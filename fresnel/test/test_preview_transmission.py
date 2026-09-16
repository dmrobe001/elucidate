# Copyright (c) 2016-2026 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Test what ``tracer.Preview`` makes of a transmissive material.

Preview shades one hit against the lights and traces no path, so it cannot
reproduce the path tracer's reflect-or-refract split or its Beer-Lambert
absorption. It takes the simplest reading of the material that still shows
refraction: a surface with any ``spec_trans`` at all is seen straight through,
bent by ``ior``, and tinted by ``color`` once per interface crossed.
"""

import fresnel
import numpy
import pytest

import conftest


def _sphere_scene(
    device, ior=1.0, spec_trans=1.0, color=1.0, background=1.0, alpha=1.0, solid=0.0
):
    """Build a scene holding one sphere against a flat background."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())
    scene.background_color = fresnel.color.linear([background] * 3)
    scene.background_alpha = alpha

    sphere = fresnel.geometry.Sphere(scene, position=[[0, 0, 0]], radius=1.0)
    sphere.material = fresnel.material.Material(
        color=fresnel.color.linear([color] * 3),
        spec_trans=spec_trans,
        ior=ior,
        solid=solid,
        roughness=0.0,
    )

    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 5), look_at=(0, 0, 0), up=(0, 1, 0), height=4
    )

    return scene


def _banded_scene(device, ior=1.0, color=1.0, with_sphere=True):
    """Build a scene holding one sphere in front of a red bar over a blue one.

    The bars are far enough behind the sphere, and thick enough, that the
    whole of the sphere's interior looks out at one or the other of them.
    """
    scene = fresnel.Scene(device, lights=conftest.test_lights())
    scene.background_color = fresnel.color.linear([0.0] * 3)
    scene.background_alpha = 1.0

    bars = fresnel.geometry.Cylinder(scene, N=2)
    bars.points[:] = [[[-8, 1.6, -6], [8, 1.6, -6]], [[-8, -1.6, -6], [8, -1.6, -6]]]
    bars.radius[:] = [1.4, 1.4]
    red = fresnel.color.linear([1.0, 0.0, 0.0])
    blue = fresnel.color.linear([0.0, 0.0, 1.0])
    bars.color[:] = [[red, red], [blue, blue]]
    bars.material = fresnel.material.Material(solid=1.0, primitive_color_mix=1.0)

    if with_sphere:
        sphere = fresnel.geometry.Sphere(scene, position=[[0, 0, 0]], radius=1.0)
        sphere.material = fresnel.material.Material(
            color=fresnel.color.linear([color] * 3),
            spec_trans=1.0,
            ior=ior,
            roughness=0.0,
        )

    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 6), look_at=(0, 0, 0), up=(0, 1, 0), height=4
    )

    return scene


def _preview(scene):
    """Render a scene with the preview tracer, without anti-aliasing."""
    return fresnel.preview(scene, w=64, h=64, anti_alias=False)[:]


# ---------------------------------------------------------------------------
# seeing through
# ---------------------------------------------------------------------------


def test_clear_material_is_invisible(device_):
    """A transmissive material with nothing to bend or absorb leaves no mark.

    At ``ior`` 1 there is no interface, and a white ``color`` absorbs nothing,
    so a preview of the sphere has to come out pixel for pixel identical to a
    preview of the scene without it. This is the preview counterpart of
    ``test_refraction.test_backdrop_visible_through_glass``, and it is the
    whole of the change: Preview used to shade this sphere as opaque plastic.
    """
    with_sphere = _preview(_banded_scene(device_, ior=1.0))
    without_sphere = _preview(_banded_scene(device_, with_sphere=False))

    numpy.testing.assert_array_equal(with_sphere, without_sphere)


def test_opaque_material_is_unchanged(device_):
    """Nothing about an opaque material's shading moved.

    ``spec_trans`` 0 is the whole of the reference image suite, so the opaque
    branch has to be untouched.
    """
    buf = _preview(_sphere_scene(device_, spec_trans=0.0, color=0.5))

    # the sphere is lit and sits against the background rather than vanishing
    assert buf[32, 32, 0:3].max() > 0
    assert not numpy.array_equal(buf[32, 32, 0:3], buf[2, 2, 0:3])


def test_any_transmission_is_full_transmission(device_):
    """Preview does not blend an opaque lobe with a transmissive one.

    There is one shading sample per hit and no way to split it between two
    lobes, so a partly transmissive material is read as fully transparent
    rather than as a mixture.
    """
    half = _preview(_sphere_scene(device_, spec_trans=0.5))
    full = _preview(_sphere_scene(device_, spec_trans=1.0))

    numpy.testing.assert_array_equal(half, full)


def test_solid_material_stays_opaque(device_):
    """``solid`` wins over ``spec_trans``, as it does in the path tracer.

    ``path_tracer_hit`` tests ``isSolid()`` first and turns the surface into
    an emitter; a solid material is a flat color, not a window.
    """
    buf = _preview(_sphere_scene(device_, solid=1.0, color=0.5, background=1.0))

    # the flat color, not the background behind it
    assert buf[32, 32, 0] == 128
    assert buf[2, 2, 0] == 255


def test_outline_stays_opaque(device_):
    """An outline is a separate material, and this one is opaque.

    Outlining a glass object is the reason to keep the outline material's own
    ``spec_trans`` in charge of its own surface.
    """
    scene = _sphere_scene(device_)
    scene.geometry[0].outline_material = fresnel.material.Material(
        solid=1.0, color=(0, 0, 0)
    )
    scene.geometry[0].outline_width = 0.2

    buf = _preview(scene)

    # a black rim against a white background and a transparent interior
    assert (buf[:, :, 0:3].sum(axis=2) < 60).sum() > 100


# ---------------------------------------------------------------------------
# color
# ---------------------------------------------------------------------------


def test_color_tints_what_is_behind(device_):
    """The material color multiplies whatever the ray reaches through it."""
    clear = _preview(_sphere_scene(device_, color=1.0, background=1.0))
    tinted = _preview(_sphere_scene(device_, color=0.5, background=1.0))

    assert clear[32, 32, 0] == 255
    assert tinted[32, 32, 0] < clear[32, 32, 0]

    # outside the sphere both show the untinted background
    assert tinted[2, 2, 0] == 255


def test_color_applies_once_per_interface(device_):
    """A closed solid picks up its color twice, entering and leaving.

    This is the preview's stand-in for Beer-Lambert absorption, which needs a
    distance travelled inside the material that a preview never computes. The
    consequence is that Preview and Path agree on what is see-through but not
    on how deep its color is.
    """
    color = 0.5
    buf = _preview(_sphere_scene(device_, color=color, background=1.0))

    # a ray down the middle of the sphere crosses two interfaces at normal
    # incidence, so it is the background times the linear color squared
    measured = fresnel.color.linear(buf[32, 32, 0:3] / 255.0)
    expected = fresnel.color.linear([color] * 3) ** 2

    numpy.testing.assert_allclose(measured, expected, atol=0.005)


def test_tint_accumulates_over_several_objects(device_):
    """Stacking transparent objects darkens what is behind them."""
    single = _preview(_sphere_scene(device_, color=0.8, background=1.0))

    stacked = _sphere_scene(device_, color=0.8, background=1.0)
    behind = fresnel.geometry.Sphere(stacked, position=[[0, 0, -2.5]], radius=0.9)
    behind.material = fresnel.material.Material(
        color=fresnel.color.linear([0.8] * 3), spec_trans=1.0, ior=1.0, roughness=0.0
    )

    assert _preview(stacked)[32, 32, 0] < single[32, 32, 0]


def test_alpha_passes_through(device_):
    """A transparent object does not cover the background.

    Preview marked every hit as fully opaque, so a glass object punched a
    solid hole in an otherwise transparent background.
    """
    transparent = _preview(_sphere_scene(device_, spec_trans=1.0, alpha=0.0))
    opaque = _preview(_sphere_scene(device_, spec_trans=0.0, alpha=0.0))

    assert transparent[32, 32, 3] == 0
    assert opaque[32, 32, 3] == 255


# ---------------------------------------------------------------------------
# index of refraction
# ---------------------------------------------------------------------------


def test_ior_bends_the_view(device_):
    """A solid glass sphere is a ball lens, and inverts what is behind it.

    The bars run left to right, red above blue. Seen through the sphere they
    swap over, which is a directional check on the refraction and not just on
    the image having changed: getting the sign of the bend wrong leaves them
    the right way up.
    """
    backdrop = _preview(_banded_scene(device_, with_sphere=False))
    lens = _preview(_banded_scene(device_, ior=1.5))

    # above the middle: red directly, blue through the lens
    assert backdrop[24, 32, 0] == 255 and backdrop[24, 32, 2] == 0
    assert lens[24, 32, 0] == 0 and lens[24, 32, 2] == 255

    # below the middle: the other way around
    assert backdrop[40, 32, 0] == 0 and backdrop[40, 32, 2] == 255
    assert lens[40, 32, 0] == 255 and lens[40, 32, 2] == 0


@pytest.mark.parametrize("ior", [1.0, 1.33, 1.5, 2.4])
def test_clear_sphere_over_a_flat_background_loses_no_light(device_, ior):
    """Bending light does not destroy it.

    Nothing reflects off a transparent interface here, and a sphere refracts a
    ray back out parallel to the way it came in, so a lossless sphere in front
    of a uniform background has to disappear whatever its ``ior``. This is the
    preview's white furnace test: it catches a tint applied to a ray that only
    reflected, and a crossing budget spent before the ray got back out.
    """
    buf = _preview(_sphere_scene(device_, ior=ior, color=1.0, background=1.0))

    assert buf[:, :, 0:3].min() == 255


def test_ior_one_does_not_bend(device_):
    """Matched indices are not an interface.

    ``refract()`` takes this case before the critical angle test, so a grazing
    ray must not be turned back as though it had totally internally reflected.
    """
    flat = _preview(_banded_scene(device_, ior=1.0))
    backdrop = _preview(_banded_scene(device_, with_sphere=False))

    numpy.testing.assert_array_equal(flat, backdrop)


def test_many_crossings_terminate(device_):
    """A ray cannot be followed forever.

    Preview has no Russian roulette, so ``direct_tracer_max_crossings`` is
    what bounds a ray that keeps finding transparent surfaces. Past the budget
    the surface shades as though it were opaque, which is bounded and visible
    rather than a black pixel or a hang.
    """
    scene = fresnel.Scene(device_, lights=conftest.test_lights())
    scene.background_color = fresnel.color.linear([1.0] * 3)
    scene.background_alpha = 1.0

    # 20 nested crossings, more than the budget allows
    glass = fresnel.geometry.Sphere(
        scene, position=[[0, 0, -2.0 * k] for k in range(20)], radius=0.9
    )
    glass.material = fresnel.material.Material(
        color=fresnel.color.linear([1.0] * 3), spec_trans=1.0, ior=1.0, roughness=0.0
    )

    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 5), look_at=(0, 0, -19), up=(0, 1, 0), height=4
    )

    buf = _preview(scene)

    # the ray ran out of crossings on a lit surface rather than on nothing
    assert buf[32, 32, 0:3].max() > 0
    assert buf[32, 32, 3] == 255
