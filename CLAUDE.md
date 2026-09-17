# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a fork of [glotzerlab/fresnel](https://github.com/glotzerlab/fresnel) that adds
refraction, volumetric absorption and rough transmission. Changes are not sent upstream.

## Commands

On Claude Code on the web, `.claude/hooks/session-start.sh` has already installed the
dependencies, built the project, and put `.venv/bin` on `PATH` with `PYTHONPATH` pointing at
`build/`. Locally, follow `BUILDING.rst`, which uses pixi.

```bash
ninja -C build                        # rebuild after editing C++
cd build && python -m pytest fresnel/test -q          # full suite, ~2.5 minutes
cd build && python -m pytest fresnel/test/test_refraction.py -q      # one module
cd build && python -m pytest fresnel/test -q -k test_white_furnace   # one test
ruff check fresnel/ && ruff format fresnel/           # Python lint and format
clang-format --style=file -i <file>                   # C++ format
```

The session hook configures the CPU backend only. To build the CUDA backend as well, configure a
second tree in `build-cuda/`; nvcc needs a host compiler it supports, so pass
`-DCMAKE_CUDA_HOST_COMPILER=` when the default `g++` is too new.

```bash
cmake -B build-cuda -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_HOST_COMPILER=g++-12
ninja -C build-cuda
```

Editing a Python file under `fresnel/` requires `ninja -C build` too: CMake copies the
sources into the build tree, and pytest runs against the copies.

## Architecture

### The backend split is load-bearing

`fresnel/common/` is backend-neutral, header-only, and annotated with a `DEVICE` macro that
expands to `__host__ __device__` under nvcc and to nothing on the host. It holds the camera,
the RNG, the materials, every ray-primitive intersection routine, and the entire path tracer
shading kernel (`path_tracer_hit` and `path_tracer_miss` in `TracerPathMethods.h`).

`fresnel/cpu/` supplies only what a backend must: acceleration structure (Embree), buffer
management, the per-pixel driver loop, and the pybind11 glue. `fresnel/gpu/` supplies the same
four things for CUDA: a Morton code LBVH in place of Embree, managed memory buffers, a megakernel
per tracer, and its own pybind11 glue. It is built only when `ENABLE_CUDA=ON`, which is off by
default.

**Do not move backend-neutral logic into `fresnel/cpu/` or `fresnel/gpu/`.** Everything in
`common/` is what both backends reuse unchanged. Anything in `common/` must stay compilable for
the device: keep `Material` a POD, keep new functions `DEVICE`-annotated, and avoid host-only
headers.

`fresnel/gpu/` has one structural rule of its own: `TracerMethods.h` and `BVHBuild.h` hold the
per-thread work as `DEVICE` functions and the `.cu` files are thin launch wrappers around them.
That is what lets the device code be driven from the host, which is the only way to exercise it
on a machine without a GPU. Do not inline those bodies back into the kernels.

Because `common/RayGen.h` reaches Random123's `boxmuller.hpp`, which takes `CUDART_VERSION` to
mean the CUDA math library will supply `sincospif`, any translation unit that includes both
`cuda_runtime.h` and `RayGen.h` must be compiled by nvcc. That is why `module-gpu.cc` includes
the tracer class headers but not `TracerMethods.h`.

### Two tracers, different capabilities

`tracer.Preview` (`TracerDirect`) shades one hit against the lights analytically and casts no
shadow rays. `tracer.Path` (`TracerPath`) is the path tracer.

Both see through a transmissive material, but only the path tracer models one. `Preview`
spends its one shading sample per hit on the surface it ends up on, so it cannot split that
sample between an opaque lobe and a dielectric one: any `spec_trans` above 0 is fully
transparent, the interface is smooth whatever the `roughness`, nothing reflects off it, and
the color is `color` applied once per interface crossed rather than Beer-Lambert absorption
over `transmission_distance`. `ior` bends light identically in both. The secondary rays
`Preview` does trace are the continuation of the primary ray through those interfaces and
nothing else — it still traces no scattered rays.

The consequence to keep in mind is that `Preview` and `Path` agree on *what is see-through*
and on *where it bends light to*, and deliberately disagree on the depth of its color. This
is documented on `Material` and on `tracer.Preview`, and is not a bug.

`emission` divides them the same way. Both show an emissive surface glowing, at the same
radiance, but only `Path` shows the light it casts on anything else, because that takes a
scattered ray.

The per-hit body of the direct tracer lives in `common/TracerDirectMethods.h`, which is to
`Preview` what `common/TracerPathMethods.h` is to `Path`. Only the traversal around it —
`rtcIntersect1()` on the CPU, `bvh_intersect()` on the GPU — belongs to a backend, so a change
to preview shading belongs in the common header and lands on both backends at once.

### Normals are geometric, never ray-facing

Intersection routines report the normal pointing out of the primitive whichever side the ray
struck. Each tracer flips it to the viewing side once, before shading. Do not reintroduce a
per-geometry flip: refraction depends on being able to tell an entry hit from an exit hit, and
that test is `dot(Ng, ray_direction) > 0`, derived rather than carried as a hit attribute.

### The hit contract

Embree carries two fresnel-specific attributes out of an intersection alongside the standard
normal and distance, in `FresnelRTCIntersectContext` (`cpu/embree_platform.h`):

- `d` — distance from the hit to the nearest primitive edge, for outline rendering
- `shading_color` — per-primitive, per-face or barycentrically interpolated colour

The GPU backend carries the same two in `HitInfo` (`gpu/cuda_platform.h`), which the traversal
fills in directly since there is no ray query context to piggyback on.

### The material model

`spec_trans` selects between an opaque lobe and a smooth dielectric interface. Within the
dielectric lobe, the Fresnel reflectance chooses reflection or refraction. Both selections are
made with the same probability as the lobe's weight, so the selection probability cancels and
neither branch carries a compensating factor — if you add a lobe, preserve that property or
the weights will be wrong.

Colour in a transmissive material comes from Beer-Lambert absorption along the path inside it,
applied on arriving at a back face, not from tinting each surface crossing. `roughness` drives
transmission as well as reflection.

`emission` is added at a hit and is the one term that does not go through a lobe: it does not
depend on the lights, the view direction or the BRDF, and the surface goes on scattering after
it. Both tracers add it, both add it on either face, and neither clamps it. In the path tracer
it lands after the Beer-Lambert absorption of the segment just travelled, so the far wall of an
emissive solid is seen through that solid's interior. There is no next event estimation here,
so an emissive surface lights the scene only by being hit - the same way `light.Light` already
works, which is why emission needed no new sampling machinery and left every reference image
alone.

`Material::alphaGGX()` floors the GGX roughness for the **opaque** lobe only. The distribution
is 0/0 as alpha goes to zero at normal incidence, and the NaN spreads through the image. The
dielectric lobe does not use it: visible-normal sampling degenerates cleanly to a sharp
interface, which is how `roughness=0` gives clear glass.

## Testing

`fresnel/test/reference/*.png` are rendered references compared per-pixel with a tolerance.
They are the regression signal for the whole renderer, so **any change that alters sampling
alters every image**. That makes such a change a deliberate decision, not an incidental
optimisation. Two known examples, both currently declined:

- switching the opaque lobe's `importanceSampleGGX` to visible-normal sampling
- treating the background as an environment light rather than a backdrop

The RNG is counter-based and keyed on `(pixel, seed, depth, sample)`, never on traversal
order, so renders are reproducible across runs and machines. Preserve this: it is what lets the
GPU backend be validated against the reference images.

`conftest.py` appends `("gpu", 1)` to the device fixture when `"gpu" in
fresnel.Device.available_modes`, so a CUDA build runs the whole suite against both backends with
no test changes. Where the two backends can legitimately disagree is an exact tie: two surfaces
the same distance from the ray, where Embree and the LBVH visit the candidates in a different
order. That is what the tolerance in `assert_image_approx_equal` absorbs.

Tests added for the transmission work assert physical properties rather than snapshots —
Fresnel at normal incidence, total internal reflection, the white furnace test, absorption
being exponential in distance. Where a threshold was chosen empirically, the measured values
on both sides of it are recorded in the test docstring; update them if you change the
threshold.

## Style

`prek run --all-files` runs the same checks as CI (ruff for Python, clang-format for C++).
The C++ style is unusual: braces are indented to the level of the block they open, per
`.clang-format`. Let the formatter do it rather than matching by hand.
