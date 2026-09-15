# Copyright (c) 2016-2026 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Test rough transmission through a microfacet dielectric.

A transmissive surface scatters about a microfacet normal rather than the
surface normal, so ``roughness`` frosts it: light still refracts, but about a
facet drawn from the distribution of visible normals rather than about the
surface itself. At roughness 0 the only visible facet is the surface and the
interface is a sharp mirror or a sharp refraction.
"""

import math

import fresnel
import numpy
import pytest

import conftest

ROUGHNESS = (0.0, 0.05, 0.15, 0.3, 0.5, 0.8)

# Both measurements below render a scene, and several tests want the same
# roughness series, so keep what has already been rendered.
_CACHE = {}


def _measure(fn, device, roughness):
    """Return fn(device, roughness), rendering it only once."""
    key = (fn.__name__, id(device), roughness)
    if key not in _CACHE:
        _CACHE[key] = fn(device, roughness)
    return _CACHE[key]


def furnace(device, roughness):
    """Memoized white furnace measurement."""
    return _measure(_furnace, device, roughness)


def contrast(device, roughness):
    """Memoized stripe contrast measurement."""
    return _measure(_stripe_contrast, device, roughness)


def _slab(scene, roughness, ior=1.5):
    """Attach a non-absorbing transmissive cube spanning [-1, 1]^3 to a scene."""
    origins = []
    normals = []
    for v in [-1, 1]:
        origins.extend([[v, 0, 0], [0, v, 0], [0, 0, v]])
        normals.extend([[v, 0, 0], [0, v, 0], [0, 0, v]])

    geometry = fresnel.geometry.ConvexPolyhedron(
        scene,
        {
            "face_normal": normals,
            "face_origin": origins,
            "radius": math.sqrt(3),
            "face_color": [[0.0, 0.0, 0.0]] * 6,
        },
        position=[[0, 0, 0]],
    )
    geometry.material = fresnel.material.Material(
        color=(1.0, 1.0, 1.0),
        spec_trans=1.0,
        ior=ior,
        roughness=roughness,
        # absorb nothing, so that whatever leaves the slab is only the interface
        transmission_distance=1e6,
    )
    return geometry


def _furnace(device, roughness, samples=256):
    """Return the mean linear value leaving a slab in a uniform white environment."""
    scene = fresnel.Scene(device, lights=[])
    scene.background_color = (1.0, 1.0, 1.0)
    scene.background_alpha = 1.0

    _slab(scene, roughness)

    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 5), look_at=(0, 0, 0), up=(0, 1, 0), height=1
    )

    tracer = fresnel.tracer.Path(device=device, w=32, h=32)
    tracer.sample(scene, samples=samples, light_samples=8)

    linear = numpy.array(tracer.linear_output[:])
    return linear[8:24, 8:24, 0:3].mean()


def _stripe_contrast(device, roughness, samples=256):
    """Return the contrast of a striped backdrop seen through the slab."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())
    scene.background_color = (1.0, 1.0, 1.0)
    scene.background_alpha = 1.0

    bars = fresnel.geometry.Cylinder(
        scene,
        points=[[[-4, y * 0.25, -1.5], [4, y * 0.25, -1.5]] for y in range(-12, 13)],
        radius=0.06,
    )
    bars.material = fresnel.material.Material(solid=1.0, color=(0.0, 0.0, 0.0))

    _slab(scene, roughness)

    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 6), look_at=(0, 0, 0), up=(0, 1, 0), height=1.6
    )

    tracer = fresnel.tracer.Path(device=device, w=64, h=64)
    tracer.sample(scene, samples=samples, light_samples=8)

    linear = numpy.array(tracer.linear_output[:])
    return linear[16:48, 16:48, 0:3].reshape(-1, 3).mean(axis=1).std()


def test_roughness_frosts_transmission(device_):
    """Roughness blurs what is seen through a transmissive material.

    A striped backdrop behind a smooth slab comes through at full contrast and
    washes out as the interface roughens. Measured standard deviation across
    the stripes, for the roughness values above:
    0.42, 0.41, 0.33, 0.090, 0.021, 0.018.
    """
    sharpness = [contrast(device_, r) for r in ROUGHNESS]

    # sharp when smooth
    assert sharpness[0] > 0.3

    # washed out when rough
    assert sharpness[-1] < 0.05

    # and monotone in between
    assert numpy.all(numpy.diff(sharpness) <= 0.0)


def test_smooth_transmission_stays_sharp(device_):
    """A barely rough interface is still essentially sharp.

    Roughness has to earn its blur: a value of 0.05 should not visibly frost a
    material, or every scene that left roughness at its default would soften.
    """
    sharp = contrast(device_, 0.0)
    nearly_sharp = contrast(device_, 0.05)

    assert nearly_sharp == pytest.approx(sharp, rel=0.05)


def test_white_furnace(device_):
    """A non-absorbing object in a uniform white environment is invisible.

    Every path through the slab either reflects or refracts, and in an
    environment of uniform radiance 1 both end up looking at the same white, so
    a lossless interface has to hand back exactly what it received. This is the
    standard check that a BSDF neither invents nor loses energy.
    """
    assert furnace(device_, 0.0) == pytest.approx(1.0, abs=0.01)
    assert furnace(device_, 0.05) == pytest.approx(1.0, abs=0.01)


def test_transmission_never_gains_energy(device_):
    """No roughness makes the interface emit light of its own.

    The furnace estimator converges on 1 from slightly above, so the bound
    allows for the Monte Carlo noise left at this sample count rather than
    sitting exactly on 1.
    """
    for roughness in ROUGHNESS:
        assert furnace(device_, roughness) <= 1.005


def test_single_scattering_loses_energy_when_rough(device_):
    """Energy loss at high roughness is expected, and is a known limitation.

    This models light striking one microfacet and leaving. Light that would
    have bounced between facets is dropped, so a rough interface returns less
    than it received, increasingly so as it roughens. Recovering it needs
    multiple scattering compensation, which this does not implement.

    Measured furnace values, for the roughness values above:
    1.001, 1.001, 1.000, 0.991, 0.913, 0.582.
    """
    energy = [furnace(device_, r) for r in ROUGHNESS]

    assert numpy.all(numpy.diff(energy) <= 0.0)
    assert energy[-1] < 0.8


def test_roughness_zero_is_a_sharp_interface(device_):
    """At roughness 0 the only visible facet is the surface normal itself.

    The microfacet machinery has to collapse exactly, not approximately, or a
    material asking for clear glass would get a faint haze.
    """
    assert furnace(device_, 0.0) == pytest.approx(1.0, abs=0.01)
    assert contrast(device_, 0.0) > 0.3
