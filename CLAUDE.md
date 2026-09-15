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

Editing a Python file under `fresnel/` requires `ninja -C build` too: CMake copies the
sources into the build tree, and pytest runs against the copies.

## Architecture

### The backend split is load-bearing

`fresnel/common/` is backend-neutral, header-only, and annotated with a `DEVICE` macro that
expands to `__host__ __device__` under nvcc and to nothing on the host. It holds the camera,
the RNG, the materials, every ray-primitive intersection routine, and the entire path tracer
shading kernel (`path_tracer_hit` and `path_tracer_miss` in `TracerPathMethods.h`).

`fresnel/cpu/` supplies only what a backend must: acceleration structure (Embree), buffer
management, the per-pixel driver loop, and the pybind11 glue.

**Do not move backend-neutral logic into `fresnel/cpu/`.** A GPU backend is planned (see the
open issue on the CUDA backend) and everything in `common/` is what it will reuse unchanged.
Anything in `common/` must stay compilable for the device: keep `Material` a POD, keep new
functions `DEVICE`-annotated, and avoid host-only headers.

### Two tracers, different capabilities

`tracer.Preview` (`TracerDirect`) shades one hit against the lights analytically, traces no
secondary rays, and casts no shadow rays. `tracer.Path` (`TracerPath`) is the path tracer.
Transmission, refraction and absorption only exist in the path tracer; `Preview` renders a
transmissive material as opaque. This is documented on `Material` and is not a bug.

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

### The material model

`spec_trans` selects between an opaque lobe and a smooth dielectric interface. Within the
dielectric lobe, the Fresnel reflectance chooses reflection or refraction. Both selections are
made with the same probability as the lobe's weight, so the selection probability cancels and
neither branch carries a compensating factor — if you add a lobe, preserve that property or
the weights will be wrong.

Colour in a transmissive material comes from Beer-Lambert absorption along the path inside it,
applied on arriving at a back face, not from tinting each surface crossing. `roughness` drives
transmission as well as reflection.

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
order, so renders are reproducible across runs and machines. Preserve this: it is what lets a
second backend be validated against the reference images.

Tests added for the transmission work assert physical properties rather than snapshots —
Fresnel at normal incidence, total internal reflection, the white furnace test, absorption
being exponential in distance. Where a threshold was chosen empirically, the measured values
on both sides of it are recorded in the test docstring; update them if you change the
threshold.

## Style

`prek run --all-files` runs the same checks as CI (ruff for Python, clang-format for C++).
The C++ style is unusual: braces are indented to the level of the block they open, per
`.clang-format`. Let the formatter do it rather than matching by hand.
