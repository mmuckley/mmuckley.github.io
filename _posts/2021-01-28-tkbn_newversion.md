---
title: "Updating torchkbnufft to 1.0: Overview of Improvements"
excerpt: "Today I am happy to announce the relase of version 1.0 of `torchkbnufft`. There are many changes: complex number support, an improved backend, a better density compensation function, and more detailed documentation."
---

## Introduction

Today I am happy to announce the relase of version 1.0 of `torchkbnufft` ([GitHub](https://github.com/mmuckley/torchkbnufft), [Documentation](https://torchkbnufft.readthedocs.io/en/stable/)). There are many changes: complex number support, an improved backend that is 4 times faster on the CPU and 2 times faster on the GPU, a better density compensation function, and more detailed documentation. Why all the updates now? Well, recently PyTorch began supporting complex tensors natively (you can read about complex number support [here](https://github.com/pytorch/pytorch/issues/33152)). Before, we had to use four high-level multiplies in Python for complex multiplications. With native complex tensor support, we can move these multiplications down to lower-level PyTorch code for a significant speed-up. This, along with updates for the PyTorch FFT API, prompted a rewrite of `torchkbnufft`.

In updating the code for complex multiplications I noticed many other areas for improvement. In this post I'll document some of the more important ones as well as my reasoning for making the changes.

## An Updated API

The first thing that most users will notice with `torchkbnufft` version 1.0 is a different API. Previously, for an MRI problem with a batch size of 5, 8 channels, and height/width of 64, you would pass a tensor to the forward NUFFT of shape `[5, 8, 2, 64, 64]`, where the `2` dimension was the real/imaginary dimension. This was always a little bit strange - PyTorch's FFT expected the real/imaginary dimension to be at the end of the shape. The reason for this was that a lot of early deep learning MRI models would include real/imaginary in the channel dimension for convolutions. However, for version 1.0, we decided to convert the NUFFT to follow PyTorch FFT convention. There are a couple of reasons for this: 1) it brings us in line with the PyTorch ecosystem and 2) it's very easy to convert real tensors (with last dimension of size `2`) to complex tensors. So now the package does this for any real input and we can have a more efficient backend based on complex tensors.

As a result, for our problem above, you'll now be expected to pass in a complex-valued tensor with shape `[5, 8, 64, 64]`. You can still pass in a real tensor with separate real/imaginary dimensions as `[5, 8, 64, 64, 2]`, but for the NUFFT code it will be converted to complex and then back to real before returning to you. Not having to deal with real tensors in the backend simplifies the code base and makes things more efficient.

Another change that people will notice is the size of the k-space trajectory. Previously, it would have been `[5, 2, klength]`, where `klength` was the number of k-space samples. The idea was that you could apply a different k-space trajectory for each batch element. In the end, I decided to remove this feature and only do one k-space trajectory for a forward pass. The reason is that in the underlying code, I just wrote a `for` loop over the different trajectories. This took away some optimization opportunities in the backend (detailed below). It's better for `torchkbnufft` to only take one trajectory for the forward pass and have the user write `for` loops over their trajectories while I write a more efficient backend, so this is the behavior in 1.0.

## Improved Indexing Operations

The slowest part of `torchkbnufft` are its indexing operations. These are pretty difficult to handle in a high-level library, and the solutions that I have at the moment still may not be ideal. Nonetheless, for version 1.0 we managed to make some improvements over what the package did previously, achieving about a four-fold speedup for forward/backward on the CPU and a two-fold speedup on the GPU. For all the pseudo-code I show below, you can see the full, up-to-date version [on GitHub](https://github.com/mmuckley/torchkbnufft/blob/master/torchkbnufft/_nufft/interp.py). Prior to version 1.0, the indexing operation for the forward interpolation looked like this:

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

The code calculates `coef`, which are interpolation coefficients based on the Kaiser-Bessel kernel, and `arr_ind`, which are the indices of the neighbors to use for interpolation. The key indexing operation is `torch.gather(griddat, 2, arr_ind)`. The GPU implementation in 1.0 is basically the same, but uses complex numbers for multiplication and `griddat[:, :, arr_ind]` instead of `torch.gather`. I'll focus on the larger changes for the CPU version.

The primary issue with this code on the CPU is that [indexing into an array is slow in PyTorch](https://github.com/pytorch/pytorch/issues/29973). We can mitigate this by minimizing the size of the index problem - in version 1.0 of `torchkbnufft`, we split up the k-space trajectory and send a different chunk of the trajectory to each process as follows:

```python
@torch.jit.script
def table_interp_over_batches(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    num_forks: int,
) -> Tensor:
    """Table interpolation backend (see table_interp())."""

    # indexing is worse when we have repeated indices - let's spread them out
    klength = omega.shape[1]
    omega_chunks = [omega[:, ind:klength:num_forks] for ind in range(num_forks)]

    futures: List[torch.jit.Future[torch.Tensor]] = []
    for omega_chunk in omega_chunks:
        futures.append(
            torch.jit.fork(
                table_interp_one_batch,
                image,
                omega_chunk,
                tables,
                n_shift,
                numpoints,
                table_oversamp,
                offsets,
            )
        )

    kdat = torch.zeros(
        image.shape[0],
        image.shape[1],
        omega.shape[1],
        dtype=image.dtype,
        device=image.device,
    )

    for ind, future in enumerate(futures):
        kdat[:, :, ind:klength:num_forks] = torch.jit.wait(future)

    return kdat
```

In this case, `table_interp_one_batch` is basically the same as our old table interpolation function. The forks will execute asynchronously over their separate k-space chunks using `torch.jit.fork` (see [here](https://pytorch.org/docs/stable/generated/torch.jit.fork.html)), and at the end we'll join them all together and return. This speeds up indexing operations by reducing the number of k-space points to look at and is one of the main sources of our improvements.

We've also changed the adjoint, where we have to scatter a k-space trajectory on to an equispaced grid using the Kaiser-Bessel kernel. Prior to 1.0, it looked like this:

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

You might notice the device branch. For some reason, `index_put_` with `accumulate=True` was faster on the CPU, whereas `index_add_` was faster on the GPU. I haven't observed this anymore when building PyTorch off its master branch, so we'll probably use `index_add_` for everything going forward once the next version of PyTorch is out.

The issue with the old code for the adjoint is that the double `for` loop over batch and real/imaginary indices isn't very fast on the CPU branch. Furthermore, `index_add_` doesn't work very well for the GPU branch over batch dimensions, either. It would be better to dispatch a bunch of workers to work on every independent batch and coil element, and this is exactly what 1.0 does. The code I'm showing below is a partial construction of how we now do adjoint interpolation showing the key pieces.

```python
def accum_tensor_index_add(image: Tensor, arr_ind: Tensor, data: Tensor) -> Tensor:
    """We fork this function for the adjoint accumulation."""
    return image.index_add_(0, arr_ind, data)


def accum_tensor_index_put(image: Tensor, arr_ind: Tensor, data: Tensor) -> Tensor:
    """We fork this function for the adjoint accumulation."""
    return image.index_put_((arr_ind,), data, accumulate=True)


@torch.jit.script
def fork_and_accum(image: Tensor, arr_ind: Tensor, data: Tensor, num_forks: int):
    device = image.device

    futures: List[torch.jit.Future[torch.Tensor]] = []
    for batch_ind in range(image.shape[0]):
        for coil_ind in range(image.shape[1]):
            # if we've used all our forks, wait for one to finish and pop
            if len(futures) == num_forks:
                torch.jit.wait(futures[0])
                futures.pop(0)

            # one of these is faster on cpu, other is faster on gpu
            if device == torch.device("cpu"):
                futures.append(
                    torch.jit.fork(
                        accum_tensor_index_put,
                        image[batch_ind, coil_ind],
                        arr_ind,
                        data[batch_ind, coil_ind],
                    )
                )
            else:
                futures.append(
                    torch.jit.fork(
                        accum_tensor_index_add,
                        image[batch_ind, coil_ind],
                        arr_ind,
                        data[batch_ind, coil_ind],
                    )
                )
    _ = [torch.jit.wait(future) for future in futures]


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

tmp = coef * data
if not device == torch.device("cpu"):
    tmp = torch.view_as_real(tmp)

# this is a much faster way of doing index accumulation
if USING_OMP:
    torch.set_num_threads(threads_per_fork)
fork_and_accum(image, arr_ind, tmp, num_forks)
if USING_OMP:
    torch.set_num_threads(num_threads)
```

Using `torch.jit.fork`, we create a new asynchronous task for every batch and coil element. The tasks each handle accumulation for their own element. The calls to `torch.jit.wait` causes the code to wait for these asynchronous tasks to finish. Since the accumulation is done in-place, we don't have to worry about whatever these tasks return. However, there is one thing we have to worry about with forking: OpenMP. If we don't do a bit of thread management, then we can use more threads than we were given or suffer performance degradation from oversubscription. To prevent this, we do a little bit of thread management to make sure that we don't have too many forks.

The adjoint operation with forking is faster - more than a factor-of-4 over the previous implementation for the CPU and a factor-of-2 for the GPU. (Note: the GPU operations are still real-valued, but this should change in the future when `index_add_` supports complex numbers.)

Overall these improvements have made version 1.0 of `torchkbnufft` about four times as fast as previously on the CPU and and two times as fast on the GPU. The forward operation was bound more by the complex multiplies and indexing - we get about a 2-3 speed-up by using complex tensors and using `torch.jit.fork` to break up the trajectory. The adjoint operation was bound by the accumulation, and we get a 2-5 speedup by using `torch.jit.fork` to dispact over batches and coils.

## Scaling

The package will scale very well over coils and batch dimensions. In general, we're bound by our indexing operations, so the main thing that makes NUFFTs slower or faster is the size of the k-space trajectory.

One thing that does affect indexing is using a 3D NUFFT. The package is faster for 3D than before, but unfortunately the speedup isn't as consistent. PyTorch indexing begins to perform worse with larger arrays, and this is the situation we have for 3D NUFFTs. There are a few steps you can take that will help:

1. Use 32-bit precision instead of 64.
2. Lower the oversampling ratio.
3. Use fewer neighbors for interpolation (e.g., set `numpoints=4`).
4. Use a GPU.

But if that's not good enough, then you're running into the limitations of the package.

## Updates to Documentation

[Documentation](https://torchkbnufft.readthedocs.io/en/stable/) of `torchkbnufft` was decent on the GitHub repository with the `README.md` and several Jupyter notebooks, but the documentation on Read the Docs was a bit lacking. It only consisted of an API, and the layout of the table of contents made it hard to navigate.

This has also been updated substantially for 1.0. We now prominently display our core modules: `KbInterp`, `KbInterpAdjoint`, `KbNufft`, `KbNufftAdjoint`, and `ToepNufft`. Each one of these is now accompanied by a mathematical description of the operations as well as detailing connections to notation in [Fessler's NUFFT paper](https://doi.org/10.1109/TSP.2002.807005). (Note: If you have any comments or notice any errors in the documentation, please let me know!) We also prominently display our primary utility functions: `calc_density_compensation_function`, `calc_tensor_spmatrix`, and `calc_toeplitz_kernel`. This should make it a lot easier for beginners to navigate the package.

## A New Density Compensation Function

Thanks to a notification from Zaccharie Ramzi and an implementation by Chaithya G.R., we got a [pull request](https://github.com/mmuckley/torchkbnufft/pull/13) for implementing [Pipe's density compensation method](https://doi.org/10.1002/(SICI)1522-2594(199901)41:1%3C179::AID-MRM25%3E3.0.CO;2-V). This was quite a bit better than my original method which presumably only worked for radial trajectories. The density compensation function calculator also has a simplified interface.

## Conclusions

Version 1.0 of `torchkbnufft` was essentially a complete rewrite of the repository and its documentation. The result is a faster, better-documented NUFFT package that retains its original benefit of being written completely in high-level Python.

This remains a personal project unaffiliated with my official position at FAIR, so all of this work was done on my own time. Still, I think it was quite rewarding, and I'm happy with the improvements to the repository.

For my next project, I think it may be finally time to move beyond Python. I've grown to love Python and PyTorch over the last 2+ years, but there are so many cool languages out there to try, I think I'll have to look into one of those next...
