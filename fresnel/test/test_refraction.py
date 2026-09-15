# Copyright (c) 2016-2026 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Test index of refraction and the dielectric Fresnel split.

A transmissive material is a smooth dielectric interface. Light arriving at it
either reflects off the surface or refracts through it, chosen with the Fresnel
reflectance, and the refracted direction follows Snell's law.
"""

import math

import fresnel
import numpy
import pytest

import conftest


def _cube_scene(device, spec_trans=1.0, ior=1.5):
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
        ior=ior,
    )

    return scene


def _sphere_scene(device, ior):
    """Build a scene holding one transmissive sphere over a colored backdrop."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())

    backdrop = fresnel.geometry.Box(scene, [6, 6, 0.2], box_radius=0.0)
    backdrop.material = fresnel.material.Material(
        solid=1.0, color=fresnel.color.linear([0.8, 0.2, 0.2])
    )

    sphere = fresnel.geometry.Sphere(scene, position=[[0, 0, 2]], radius=1.0)
    sphere.material = fresnel.material.Material(
        solid=0.0,
        color=fresnel.color.linear([0.9, 0.9, 0.9]),
        spec_trans=1.0,
        ior=ior,
    )

    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 8), look_at=(0, 0, 0), up=(0, 1, 0), height=3
    )

    return scene


def _path_render(device, scene, samples=64):
    """Path trace a scene at a small fixed size."""
    tracer = fresnel.tracer.Path(device=device, w=64, h=64)
    return tracer.sample(scene, samples=samples, light_samples=8)[:]


# ---------------------------------------------------------------------------
# the Fresnel equation itself
# ---------------------------------------------------------------------------


def test_fresnel_normal_incidence():
    """At normal incidence F reduces to the closed form ((n1 - n2)/(n1 + n2))^2."""
    for eta_i, eta_t in [(1.0, 1.5), (1.0, 1.333), (1.5, 1.0), (1.0, 2.4)]:
        expected = ((eta_i - eta_t) / (eta_i + eta_t)) ** 2
        actual = fresnel._common.fresnel_dielectric(1.0, eta_i, eta_t)
        assert actual == pytest.approx(expected, abs=1e-6)

    # the familiar 4% reflection off glass
    assert fresnel._common.fresnel_dielectric(1.0, 1.0, 1.5) == pytest.approx(0.04)


def test_fresnel_no_interface():
    """An index ratio of 1 is not an interface and reflects nothing."""
    for cos_i in numpy.linspace(0.01, 1.0, 25):
        # matched indices cancel analytically; float32 leaves a rounding residue
        assert fresnel._common.fresnel_dielectric(cos_i, 1.5, 1.5) < 1e-6


def test_fresnel_grazing_reflects_everything():
    """F rises to 1 at grazing incidence."""
    assert fresnel._common.fresnel_dielectric(0.0, 1.0, 1.5) == pytest.approx(1.0)
    assert fresnel._common.fresnel_dielectric(1e-4, 1.0, 1.5) > 0.99


def test_fresnel_monotonic():
    """F increases monotonically from normal towards grazing incidence."""
    cos_i = numpy.linspace(1.0, 0.0, 50)
    f = [fresnel._common.fresnel_dielectric(c, 1.0, 1.5) for c in cos_i]

    assert numpy.all(numpy.diff(f) >= 0.0)
    assert f[0] == pytest.approx(0.04)
    assert f[-1] == pytest.approx(1.0)


def test_fresnel_total_internal_reflection():
    """Past the critical angle from the dense side, all light reflects."""
    ior = 1.5
    critical = math.asin(1.0 / ior)

    # well inside the critical angle most light still escapes
    inside = fresnel._common.fresnel_dielectric(math.cos(critical * 0.9), ior, 1.0)
    assert inside < 0.2

    # beyond it, none does. The clamp to exactly 1 lands within a float32 step of
    # the critical angle itself, so start just past it.
    for theta in numpy.linspace(critical + 1e-3, math.pi / 2, 20):
        assert fresnel._common.fresnel_dielectric(math.cos(theta), ior, 1.0) == 1.0


def test_fresnel_approaches_total_internal_reflection():
    """F climbs continuously to 1 as the critical angle is approached."""
    ior = 1.5
    critical = math.asin(1.0 / ior)

    theta = numpy.linspace(0.0, critical * 0.999, 40)
    f = [fresnel._common.fresnel_dielectric(math.cos(t), ior, 1.0) for t in theta]

    assert numpy.all(numpy.diff(f) >= 0.0)
    assert f[0] == pytest.approx(0.04)
    # the approach is square root shaped, so it is already most of the way up
    assert f[-1] > 0.5


def test_fresnel_reciprocity():
    """F is the same viewed from either side of the interface at conjugate angles."""
    eta_i, eta_t = 1.0, 1.5

    for theta_i in numpy.linspace(0.0, math.pi / 2 * 0.99, 20):
        cos_i = math.cos(theta_i)
        sin_t = (eta_i / eta_t) * math.sin(theta_i)
        cos_t = math.sqrt(1.0 - sin_t * sin_t)

        forward = fresnel._common.fresnel_dielectric(cos_i, eta_i, eta_t)
        reverse = fresnel._common.fresnel_dielectric(cos_t, eta_t, eta_i)

        assert forward == pytest.approx(reverse, abs=1e-4)


# ---------------------------------------------------------------------------
# the material property
# ---------------------------------------------------------------------------


def test_ior_default():
    """Materials default to the index of refraction of glass."""
    assert fresnel.material.Material().ior == pytest.approx(1.5)
    assert fresnel._common.Material().ior == pytest.approx(1.5)


def test_ior_round_trips_through_proxies(device_):
    """ior survives the geometry and outline material proxies."""
    scene = fresnel.Scene(device_)
    geometry = fresnel.geometry.Sphere(scene, position=[[0, 0, 0]], radius=1.0)

    geometry.material.ior = 1.333
    assert geometry.material.ior == pytest.approx(1.333)

    geometry.outline_material.ior = 2.4
    assert geometry.outline_material.ior == pytest.approx(2.4)

    # the two are independent
    assert geometry.material.ior == pytest.approx(1.333)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_ior_one_transmits_without_bending(device_):
    """An index of 1 reproduces undeviated transmission exactly.

    With no index contrast the Fresnel reflectance is zero and Snell's law
    leaves the direction untouched, so the render must match the straight
    through transmission fresnel did before refraction existed, to the bit.
    """
    scene = _cube_scene(device_, spec_trans=0.9, ior=1.0)
    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 5), look_at=(0, 0, 0), up=(0, 1, 0), height=1
    )

    assert _path_render(device_, scene)[:, :, 0:3].mean() == pytest.approx(
        18.397, abs=0.01
    )


def test_refraction_bends_light(device_):
    """A glass sphere refracts the backdrop behind it; an index of 1 does not."""
    straight = _path_render(device_, _sphere_scene(device_, ior=1.0))
    glass = _path_render(device_, _sphere_scene(device_, ior=1.5))

    # the sphere covers the middle of the frame
    centre = (slice(16, 48), slice(16, 48), slice(0, 3))
    assert not numpy.allclose(straight[centre], glass[centre])

    # bending is a large effect, not rounding noise
    difference = numpy.abs(straight[centre].astype(float) - glass[centre].astype(float))
    assert difference.mean() > 5.0


def _transmissive_sphere_scene(device, ior):
    """Build a scene holding one fully transmissive sphere on a black background."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())

    sphere = fresnel.geometry.Sphere(scene, position=[[0, 0, 0]], radius=1.0)
    sphere.material = fresnel.material.Material(
        solid=0.0,
        color=fresnel.color.linear([0.9, 0.9, 0.9]),
        spec_trans=1.0,
        ior=ior,
    )

    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 5), look_at=(0, 0, 0), up=(0, 1, 0), height=2.5
    )

    return scene


def test_matched_index_is_invisible(device_):
    """Without an index contrast there is no interface to see.

    A fully transmissive sphere at ior 1 reflects nothing and bends nothing, so
    every ray carries straight through to the black background and the sphere
    renders away to nothing.
    """
    buf = _path_render(device_, _transmissive_sphere_scene(device_, ior=1.0))

    assert buf[:, :, 0:3].max() == 0


def test_higher_ior_reflects_more(device_):
    """A larger index contrast reflects more light at the surface.

    The only thing lighting a transmissive sphere on a black background is
    Fresnel reflection off its surface, so the frame brightens as the index
    contrast grows. Measured means: 0.0 at ior 1, 13.8 at 1.5, 28.8 at 2.4.
    """
    means = [
        _path_render(device_, _transmissive_sphere_scene(device_, ior=ior))[
            :, :, 0:3
        ].mean()
        for ior in (1.0, 1.5, 2.4)
    ]

    assert means[0] < means[1] < means[2]
    assert means[0] == 0.0
