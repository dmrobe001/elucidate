# Copyright (c) 2016-2026 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Test the geometric normal convention and back face shading.

Intersection routines report the *geometric* normal: it points out of the
primitive whichever side the ray struck, and is never pre-flipped toward the
ray. The tracers flip it to the viewing side for shading, which means a hit on
the inside of a solid shades like any other hit instead of being dropped.

These tests cover the behaviour that convention enables. The reference image
tests elsewhere in this directory cover the other half of the contract: that
moving the flip out of the geometry did not change how front faces render.
"""

import math

import fresnel
import numpy

import conftest


def _cube_scene(device, spec_trans=0.0):
    """Build a scene holding a single cube spanning [-1, 1]^3 at the origin."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())

    origins = []
    normals = []
    for v in [-1, 1]:
        origins.extend([[v, 0, 0], [0, v, 0], [0, 0, v]])
        normals.extend([[v, 0, 0], [0, v, 0], [0, 0, v]])

    poly_info = {
        "face_normal": normals,
        "face_origin": origins,
        "radius": math.sqrt(3),
        "face_color": fresnel.color.linear([[0.9, 0.9, 0.9]] * 6),
    }

    geometry = fresnel.geometry.ConvexPolyhedron(scene, poly_info, position=[[0, 0, 0]])
    geometry.material = fresnel.material.Material(
        solid=0.0,
        color=fresnel.color.linear([0.1, 0.8, 0.3]),
        spec_trans=spec_trans,
    )

    return scene


def test_default_material_spec_trans():
    """A default constructed C++ Material zeroes every member.

    ``Material(color)`` used to leave ``spec_trans`` uninitialized, so the
    default material every geometry installs on construction carried an
    indeterminate transmission value.
    """
    m = fresnel._common.Material()

    assert m.solid == 0.0
    assert m.primitive_color_mix == 0.0
    assert m.spec_trans == 0.0
    assert m.metal == 0.0


def test_backface_shaded_direct(device_):
    """The direct tracer shades the inside of a solid.

    The camera sits inside the cube looking at the inside of the +z face. The
    geometry reports that face's outward normal, which points away from the
    ray, so the hit is a back face. Before back faces were flipped to the
    viewing side the BRDF returned zero here and the whole frame came out
    black.
    """
    scene = _cube_scene(device_)
    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 0), look_at=(0, 0, 1), up=(0, 1, 0), height=1
    )

    buf = fresnel.preview(scene, w=64, h=64, anti_alias=False)[:]

    # every pixel is a hit on the inside of the far face, so nothing is
    # background and nothing is black
    assert numpy.all(buf[:, :, 3] == 255)
    assert buf[:, :, 0:3].max() > 0

    # the face shades to its material color rather than to an arbitrary value
    assert buf[:, :, 1].mean() > buf[:, :, 0].mean()
    assert buf[:, :, 1].mean() > buf[:, :, 2].mean()


def test_backface_normal_matches_frontface(device_):
    """A back face shades the same as the front face of the same plane.

    Viewing the +z face from inside the cube and viewing the -z face from
    outside present the same geometry to the camera under the same lighting,
    differing only in which side of the plane was hit. They must shade
    identically: that is what it means for the flip to be a property of
    shading rather than of the geometry.
    """
    front = _cube_scene(device_)
    front.camera = fresnel.camera.Orthographic(
        position=(0, 0, 5), look_at=(0, 0, 1), up=(0, 1, 0), height=1
    )
    front_buf = fresnel.preview(front, w=64, h=64, anti_alias=False)[:]

    back = _cube_scene(device_)
    back.camera = fresnel.camera.Orthographic(
        position=(0, 0, 0), look_at=(0, 0, 1), up=(0, 1, 0), height=1
    )
    back_buf = fresnel.preview(back, w=64, h=64, anti_alias=False)[:]

    numpy.testing.assert_array_equal(front_buf[:], back_buf[:])


def _path_render_mean(device, spec_trans):
    """Path trace the cube head on and return the mean channel value."""
    scene = _cube_scene(device, spec_trans=spec_trans)
    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 5), look_at=(0, 0, 0), up=(0, 1, 0), height=1
    )

    tracer = fresnel.tracer.Path(device=device, w=64, h=64)
    return tracer.sample(scene, samples=64, light_samples=8)[:][:, :, 0:3].mean()


def test_backface_transmission_path(device_):
    """Paths that continue inside a solid are not terminated.

    With ``spec_trans`` set, a path transmits through the near face and then
    strikes the far face from the inside. Reflection at that second hit used
    to zero the attenuation and kill the path, so the interior lost every
    contribution that bounced inside the solid.

    Measured mean channel value at ``spec_trans=0.9``: 15.04 when those paths
    were killed, 18.40 once they shade. The threshold sits between the two.
    """
    assert _path_render_mean(device_, spec_trans=0.9) > 17.0
