---
title: "Updating torchkbnufft to 1.0: Overview of Improvements"
excerpt: "Today I am happy to announce the relase of version 1.0 of `torchkbnufft`. There are many changes: complex number support, an improved backend, a better density compensation function, and more detailed documentation."
---

## Introduction

Today I am happy to announce the relase of version 1.0 of [`torchkbnufft`](https://github.com/mmuckley/torchkbnufft). There are many changes: complex number support, an improved backend, a better density compensation function, and more detailed documentation. Why all the updates now? Well, recently PyTorch began supporting complex tensors natively (you can read about complex number support [here](https://github.com/pytorch/pytorch/issues/33152)). Before, we had to use four high-level multiplies in Python for complex multiplications. With native complex tensor support, we can move these multiplications down to lower-level PyTorch code for a significant speed-up. This, along with updates in PyTorch around FFT operations, prompted a rewrite of `torchkbnufft`.

In updating the code for complex multiplications I noticed many other areas for improvement. In this post I'll document some of the more important ones as well as my reasoning for making the changes.

## An Updated API

The first thing that most users will notice with `torchkbnufft` version 1.0 is a different API. Previously, for an MRI problem with a batch size of 5, 8 channels, and height/width of 64, you would pass a tensor to the forward NUFFT of shape `[5, 8, 2, 64, 64]`, where the `2` dimension was the real/imaginary dimension. This was always a little bit strange - PyTorch's FFT expected the real/imaginary dimension to be at the end of the shape. The reason for this was that a lot of early deep learning MRI models will include real/imaginary in the channel dimension for convolutions. However, for version 1.0, we decided to convert the NUFFT to follow PyTorch FFT convention. There are a couple of reasons for this: 1) it brings us in line with the PyTorch ecosystem and 2) it's very easy to convert real tensors (with last dimension of size `2`) to complex tensors. So now the package does this for any real input and we can have a more efficient backend based on complex tensors.

As a result, for our problem above, you'll now be expected to pass in a complex-valued tensor with shape `[5, 8, 64, 64]`. You can still pass in a real tensor with separate real/imaginary dimensions as `[5, 8, 64, 64, 2]`, but for the NUFFT code it will be converted to complex and then back to real before returning to you. Not having to deal with real tensors in the backend greatly simplifies the code base and makes things more efficient.

The last change that people will notice for table interpolation is the size of the k-space trajectory. Previously, it would have been `[5, 2, klength]`, where `klength` was the number of k-space samples. The idea was that you could apply a different k-space trajectory for each batch element. In the end, I decided to remove this feature and only do one k-space trajectory for a forward pass. The reason is that in the underlying code, I just wrote a `for` loop over the different trajectories. This took away some optimization opportunities in the backend (detailed below). It's better for `torchkbnufft` to only take one trajectory for the forward pass and have the user write `for` loops over their trajectories while I write a more efficient backend, so this is the behavior in 1.0.

## Improved Indexing Operations

The slowest part of `torchkbnufft` are its indexing operations. These are pretty difficult to handle in a high-level library and the solutions that I have at the moment still may not be ideal, but for version 1.0 we managed to make some improvements over what the package did previously. Prior to version 1.0, the indexing operation for the forward interpolation looked like this:

```python
coef, arr_ind = calc_coef_and_indices(
    tm, kofflist, Jlist[:, Jind], table, centers, L, dims
)

# unsqueeze coil and real/imag dimensions for on-grid indices
arr_ind = (
    arr_ind.unsqueeze(0).unsqueeze(0).expand(kdat.shape[0], kdat.shape[1], -1)
)

# gather and multiply coefficients
kdat += complex_mult(
    coef.unsqueeze(0), torch.gather(griddat, 2, arr_ind), dim=1
)
```

The code calculates `coef`, which are interpolation coefficients based on the Kaiser-Bessel kernel, and `arr_ind`, which are the indices of the neighbors to use for interpolation. The key indexing operation is `torch.gather(griddat, 2, arr_ind)`. For version 1.0, we use the following:

```python
coef, arr_ind = calc_coef_and_indices(
    tm=tm,
    base_offset=base_offset,
    offset_increments=offset,
    tables=tables,
    centers=centers,
    table_oversamp=table_oversamp,
    grid_size=grid_size,
)

# gather and multiply coefficients
kdat += coef * image[:, :, arr_ind]
```

These actually aren't that different. The code is a bit simpler, and it's faster on the CPU, but GPU users won't notice a huge change.

The greater difference is in the adjoint, where we have to scatter a k-space trajectory on to an equispaced grid using the Kaiser-Bessel kernel. Prior to 1.0, it looked like this:

```python
coef, arr_ind = calc_coef_and_indices(
    tm, kofflist, Jlist[:, Jind], table, centers, L, dims, conjcoef=True
)

# the following code takes ordered data and scatters it on to an image grid
# profiling for a 2D problem showed drastic differences in performances
# for these two implementations on cpu/gpu, but they do the same thing
if device == torch.device("cpu"):
    tmp = complex_mult(coef.unsqueeze(0), kdat, dim=1)
    for bind in range(griddat.shape[0]):
        for riind in range(griddat.shape[1]):
            griddat[bind, riind].index_put_(
                tuple(arr_ind.unsqueeze(0)), tmp[bind, riind], accumulate=True
            )
else:
    griddat.index_add_(2, arr_ind, complex_mult(coef.unsqueeze(0), kdat, dim=1))
```

You might notice the device branch. For some reason, `index_put_` with `accumulate=True` is a lot faster on the CPU, whereas `index_add_` is a lot faster on the GPU. The issue with this code is that the double `for` loop over batch and real/imaginary indices isn't very fast on the CPU branch, and `index_add_` doesn't work very well for the GPU branch over these dimensions, either. It would be better to dispatch a bunch of workers to work on every independent batch and coil element, and this is exactly what 1.0 does:

```python
def accum_tensor_index_add(image: Tensor, arr_ind: Tensor, data: Tensor) -> Tensor:
    """We fork this function for the adjoint accumulation."""
    return image.index_add_(0, arr_ind, data)


def accum_tensor_index_put(image: Tensor, arr_ind: Tensor, data: Tensor) -> Tensor:
    """We fork this function for the adjoint accumulation."""
    return image.index_put_((arr_ind,), data, accumulate=True)

...

coef, arr_ind = calc_coef_and_indices(
    tm=tm,
    base_offset=base_offset,
    offset_increments=offset,
    tables=tables,
    centers=centers,
    table_oversamp=table_oversamp,
    grid_size=grid_size,
    conjcoef=True,
)

# this is a much faster way of doing index accumulation
tmp = coef * data
if not device == torch.device("cpu"):
    tmp = torch.view_as_real(tmp)
futures: List[torch.jit.Future[torch.Tensor]] = []
for batch_ind in range(image.shape[0]):
    for coil_ind in range(image.shape[1]):
        if device == torch.device("cpu"):
            futures.append(
                torch.jit.fork(
                    accum_tensor_index_put,
                    image[batch_ind, coil_ind],
                    arr_ind,
                    tmp[batch_ind, coil_ind],
                )
            )
        else:
            futures.append(
                torch.jit.fork(
                    accum_tensor_index_add,
                    image[batch_ind, coil_ind],
                    arr_ind,
                    tmp[batch_ind, coil_ind],
                )
            )
_ = [torch.jit.wait(future) for future in futures]
```

Using `torch.jit.fork` (see [here](https://pytorch.org/docs/stable/generated/torch.jit.fork.html)), we create a new asynchronous task for every batch and coil element. The tasks each handle accumulation for their own element. The calls to `torch.jit.wait` causes the code to wait for these asynchronous tasks to finish. Since the accumulation is done in-place, we don't have to worry about whatever these tasks return. This is quite a bit faster - about 2.5 times faster on my laptop and perhaps 2 times faster on my CPU and GPU implementations. (Note: the GPU operations are still real-valued, but this should change in the future when `index_add_` supports complex numbers).

Overall these improvements have made version 1.0 of `torchkbnufft` about twice as fast as previously on both the CPU and GPU. The forward operation was bound more by the complex multiplies - we get about a factor-of-2 speed-up by using complex tensors. The adjoint operation was bound by the accumulation, and we get a factor-of-2 speed-up by using `torch.jit.fork`.

## Updates to Documentation

Documentation of `torchkbnufft` was good on the GitHub repository with the `README.md` and several Jupyter notebooks, but the documentation on Read the Docs was a bit lacking. It only consisted of an API, and the layout of the table and functions made it hard to navigate.

This has also been updated substantially for 1.0. We now prominently display our core modules: `KbInterp`, `KbInterpAdjoint`, `KbNufft`, `KbNufftAdjoint`, and `ToepNufft`. Each one of these is now accompanied by a mathematical description of the operations as well as detailing connections to notation in [Fessler's NUFFT paper](https://doi.org/10.1109/TSP.2002.807005). (Note: I didn't have a ton of time to closely curate the math - please let me know if you notice an error.) We also prominently display our primary utility functions: `calc_density_compensation_function`, `calc_tensor_spmatrix`, and `calc_toeplitz_kernel`. This should make it a lot easier for beginners to navigate the package.

## A New Density Compensation Function

Thanks to a notification from Z. Ramzi and an implementation by Chaithya G.R., we got a [pull request](https://github.com/mmuckley/torchkbnufft/pull/13) for implementing [Pipe's density compensation method](https://doi.org/10.1002/(SICI)1522-2594(199901)41:1%3C179::AID-MRM25%3E3.0.CO;2-V). This was quite a bit better than my original method which presumably only worked for radial trajectories. The density compensation function calculator also has a simplified interface.

## Conclusions

Version 1.0 of `torchkbnufft` was essentially a complete rewrite of the repository and its documentation. The result is a faster, better-documented NUFFT package that retains its original benefit of being written completely in high-level Python.

This remains a personal project unaffiliated with my official position at FAIR, so all of this work was done on my own time. Still, I think it was quite rewarding, and I'm happy with the improvements to the repository.

That being said, although I've grown to love Python over my 2+ years of using it, I think it might be time for me to learn a new language on my next personal project. At the moment, I think Rust is calling my name...
