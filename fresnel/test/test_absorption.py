# Copyright (c) 2016-2026 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Test Beer-Lambert absorption inside transmissive materials.

The interface of a transmissive material is clear. Color comes from the
material absorbing along the path inside it, so the further light travels
through a solid the deeper its color, and the material color is the color seen
through exactly one ``transmission_distance``.

The scenes here set ``ior=1`` so that the interface neither reflects nor bends,
which leaves absorption as the only thing acting on the light and makes the
expected result exact.
"""

import math

import fresnel
import numpy
import pytest

import conftest

# the slab spans [-1, 1] on the view axis
SLAB_THICKNESS = 2.0


def _slab_scene(device, color, transmission_distance, ior=1.0):
    """Build a scene holding a transmissive slab against a white background."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())
    # already linear: colors here are given directly in the linear space
    scene.background_color = (1.0, 1.0, 1.0)
    scene.background_alpha = 1.0

    origins = []
    normals = []
    for v in [-1, 1]:
        origins.extend([[v, 0, 0], [0, v, 0], [0, 0, v]])
        normals.extend([[v, 0, 0], [0, v, 0], [0, 0, v]])

    poly_info = {
        "face_normal": normals,
        "face_origin": origins,
        "radius": math.sqrt(3),
        "face_color": [[0.0, 0.0, 0.0]] * 6,
    }

    slab = fresnel.geometry.ConvexPolyhedron(scene, poly_info, position=[[0, 0, 0]])
    slab.material = fresnel.material.Material(
        color=color,
        spec_trans=1.0,
        ior=ior,
        transmission_distance=transmission_distance,
        # a sharp interface, so absorption is the only thing acting on the light
        roughness=0.0,
    )

    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 5), look_at=(0, 0, 0), up=(0, 1, 0), height=1
    )

    return scene


def _transmitted(device, color, transmission_distance, samples=1024):
    """Return the mean linear RGB transmitted straight through the slab."""
    tracer = fresnel.tracer.Path(device=device, w=32, h=32)
    tracer.sample(
        _slab_scene(device, color, transmission_distance),
        samples=samples,
        light_samples=8,
    )

    linear = numpy.array(tracer.linear_output[:])
    return linear[8:24, 8:24, 0:3].reshape(-1, 3).mean(axis=0)


def test_transmission_distance_default():
    """transmission_distance defaults to 1."""
    assert fresnel.material.Material().transmission_distance == pytest.approx(1.0)
    assert fresnel._common.Material().transmission_distance == pytest.approx(1.0)


def test_transmission_distance_round_trips_through_proxies(device_):
    """transmission_distance survives the geometry and outline material proxies."""
    scene = fresnel.Scene(device_)
    geometry = fresnel.geometry.Sphere(scene, position=[[0, 0, 0]], radius=1.0)

    geometry.material.transmission_distance = 4.0
    assert geometry.material.transmission_distance == pytest.approx(4.0)

    geometry.outline_material.transmission_distance = 0.25
    assert geometry.outline_material.transmission_distance == pytest.approx(0.25)

    assert geometry.material.transmission_distance == pytest.approx(4.0)


def test_color_is_seen_through_one_transmission_distance(device_):
    """At exactly one transmission_distance the material transmits its color.

    With a matched index there is nothing to reflect or bend, and the white
    background arrives at full strength, so the transmitted value is the
    material color itself.
    """
    color = (0.5, 0.25, 0.75)
    transmitted = _transmitted(device_, color, transmission_distance=SLAB_THICKNESS)

    numpy.testing.assert_allclose(transmitted, color, atol=0.02)


def test_absorption_is_exponential_in_distance(device_):
    """Halving transmission_distance squares the transmitted color.

    Beer-Lambert absorption is exponential in path length, so a slab that reads
    as ``color`` at one transmission_distance reads as ``color ** 2`` when the
    same thickness spans two of them.
    """
    color = numpy.array([0.6, 0.5, 0.8])

    once = _transmitted(device_, tuple(color), transmission_distance=SLAB_THICKNESS)
    twice = _transmitted(
        device_, tuple(color), transmission_distance=SLAB_THICKNESS / 2
    )

    numpy.testing.assert_allclose(once, color, atol=0.02)
    numpy.testing.assert_allclose(twice, color**2, atol=0.02)


def test_thicker_is_darker(device_):
    """Every channel absorbs more over a longer path."""
    color = (0.7, 0.55, 0.4)

    thin = _transmitted(device_, color, transmission_distance=SLAB_THICKNESS * 4)
    thick = _transmitted(device_, color, transmission_distance=SLAB_THICKNESS / 4)

    assert numpy.all(thin > thick)


def test_large_transmission_distance_absorbs_nothing(device_):
    """A material that absorbs over a vast distance is clear."""
    transmitted = _transmitted(
        device_, (0.2, 0.4, 0.6), transmission_distance=1e6, samples=256
    )

    numpy.testing.assert_allclose(transmitted, [1.0, 1.0, 1.0], atol=0.01)


def test_opaque_materials_do_not_absorb(device_):
    """Absorption belongs to the transmissive model.

    An opaque material has no interior for a path to travel through, so a hit
    on the inside of one must not pick up an absorption term.
    """
    scene = _slab_scene(device_, (0.5, 0.5, 0.5), transmission_distance=0.01)
    scene.geometry[0].material.spec_trans = 0.0

    # the camera sits inside the slab, so every hit is a back face
    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 0), look_at=(0, 0, 1), up=(0, 1, 0), height=1
    )

    buf = fresnel.preview(scene, w=32, h=32, anti_alias=False)[:]

    # a transmission_distance that short would absorb this to black if it applied
    assert buf[:, :, 0:3].max() > 0
