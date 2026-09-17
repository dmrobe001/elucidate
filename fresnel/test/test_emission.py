# Copyright (c) 2016-2026 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Test emissive materials.

``emission`` is radiance the surface gives off on its own. It is added at a hit
and does not depend on the lights, the view direction or the rest of the
shading, so a scene with no lights over a black background renders an emitter
as exactly its emission. That is what makes the assertions here exact rather
than approximate: they state physical properties - emission is additive,
linear, two sided, and absorbed by a material it shines through - instead of
comparing against a snapshot.
"""

import math

import fresnel
import numpy
import pytest

import conftest

# the slab in _slab_scene spans [-1, 1] along the view axis
SLAB_THICKNESS = 2.0


def _dark_scene(device, lit=False):
    """Build an empty scene that contributes no light of its own.

    With no lights and a black background, anything in the image got there by
    being emitted.
    """
    lights = conftest.test_lights() if lit else []
    scene = fresnel.Scene(device, lights=lights)
    scene.background_color = (0.0, 0.0, 0.0)
    scene.background_alpha = 1.0
    return scene


def _sphere_scene(device, lit=False, **material_args):
    """Build a scene holding one sphere that fills the view."""
    scene = _dark_scene(device, lit=lit)

    sphere = fresnel.geometry.Sphere(scene, position=[[0, 0, 0]], radius=1.0)
    sphere.material = fresnel.material.Material(**material_args)

    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 5), look_at=(0, 0, 0), up=(0, 1, 0), height=1
    )

    return scene


def _slab_scene(device, emission, **slab_args):
    """Build a scene with an emissive backdrop behind a transmissive slab.

    The slab spans [-1, 1]^3 and the backdrop is a large sphere well behind
    it, so a ray down the middle of the view crosses the full thickness of the
    slab before reaching the emitter.
    """
    scene = _dark_scene(device)

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
    slab.material = fresnel.material.Material(roughness=0.0, **slab_args)

    backdrop = fresnel.geometry.Sphere(scene, position=[[0, 0, -20]], radius=10.0)
    backdrop.material = fresnel.material.Material(emission=emission)

    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 5), look_at=(0, 0, 0), up=(0, 1, 0), height=1
    )

    return scene


def _render(scene, tracer, samples=16, w=24, h=24):
    """Render a scene and return its linear output.

    The linear buffer rather than the sRGB one: emission is exact in linear
    space, and quantizing to 8 bits would hide that.
    """
    if tracer == "preview":
        t = fresnel.tracer.Preview(device=scene.device, w=w, h=h, anti_alias=False)
        t.render(scene)
    else:
        t = fresnel.tracer.Path(device=scene.device, w=w, h=h)
        t.sample(scene, samples=samples, light_samples=4)

    return numpy.array(t.linear_output[:])


def _center(buf, half=4):
    """Return the mean linear RGB over the middle of an image."""
    j = buf.shape[0] // 2
    i = buf.shape[1] // 2
    middle = buf[j - half : j + half, i - half : i + half, 0:3]
    return middle.reshape(-1, 3).mean(axis=0)


# ---------------------------------------------------------------------------
# the property itself
# ---------------------------------------------------------------------------


def test_emission_defaults_to_black():
    """A material that was never told to glow does not glow.

    This is what leaves the rendered reference images alone: no sampling
    changed, and the new term is exactly zero unless it is asked for.
    """
    assert fresnel.material.Material().emission == (0.0, 0.0, 0.0)

    cpp = fresnel._common.Material()
    assert (cpp.emission.r, cpp.emission.g, cpp.emission.b) == (0.0, 0.0, 0.0)


def test_emission_round_trips_through_proxies(device_):
    """emission survives the geometry and outline material proxies."""
    scene = fresnel.Scene(device_)
    geometry = fresnel.geometry.Sphere(scene, position=[[0, 0, 0]], radius=1.0)

    geometry.material.emission = (0.25, 0.5, 0.75)
    assert geometry.material.emission == pytest.approx((0.25, 0.5, 0.75))

    geometry.outline_material.emission = (1.0, 2.0, 3.0)
    assert geometry.outline_material.emission == pytest.approx((1.0, 2.0, 3.0))

    # the outline material is a material of its own and did not take the other's
    assert geometry.material.emission == pytest.approx((0.25, 0.5, 0.75))


def test_emission_needs_three_components():
    """emission is a color, like every other color in the API."""
    with pytest.raises(ValueError):
        fresnel.material.Material(emission=(1.0, 1.0))


# ---------------------------------------------------------------------------
# what an emitter looks like
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tracer", ["preview", "path"])
def test_emitter_renders_its_own_radiance(device_, tracer):
    """An emitter in the dark renders as exactly its emission.

    Nothing else in the scene gives off light, so the pixel is the emitted
    radiance itself rather than a shaded version of it. Both tracers agree
    here, which is the point: the preview shows the glow even though it cannot
    show what the glow does to anything else.
    """
    emission = (0.25, 0.5, 0.75)
    buf = _render(_sphere_scene(device_, emission=emission), tracer)

    numpy.testing.assert_allclose(_center(buf), emission, atol=1e-6)


@pytest.mark.parametrize("tracer", ["preview", "path"])
def test_emission_is_linear(device_, tracer):
    """Twice the emission is twice the light."""
    single = _center(_render(_sphere_scene(device_, emission=(0.1, 0.2, 0.3)), tracer))
    double = _center(_render(_sphere_scene(device_, emission=(0.2, 0.4, 0.6)), tracer))

    numpy.testing.assert_allclose(double, 2.0 * single, rtol=1e-5)


@pytest.mark.parametrize("tracer", ["preview", "path"])
def test_emission_is_not_clamped(device_, tracer):
    """Emission is a radiance, so it is free to exceed 1.

    An emitter brighter than white is much of the point of having one, and it
    is what ``highlight_warning`` is for.
    """
    buf = _render(_sphere_scene(device_, emission=(4.0, 4.0, 4.0)), tracer)

    numpy.testing.assert_allclose(_center(buf), (4.0, 4.0, 4.0), atol=1e-5)


@pytest.mark.parametrize("tracer", ["preview", "path"])
def test_emission_ignores_the_rest_of_the_material(device_, tracer):
    """How a surface reflects light has no say in what it emits.

    ``roughness``, ``metal``, ``specular`` and ``color`` describe what the
    surface does to light arriving at it. Emission does not arrive; it leaves.
    """
    emission = (0.3, 0.3, 0.3)
    plain = _center(_render(_sphere_scene(device_, emission=emission), tracer))
    fancy = _center(
        _render(
            _sphere_scene(
                device_,
                emission=emission,
                color=(0.1, 0.9, 0.2),
                roughness=0.9,
                metal=1.0,
                specular=1.0,
            ),
            tracer,
        )
    )

    numpy.testing.assert_allclose(fancy, plain, atol=1e-6)


@pytest.mark.parametrize("tracer", ["preview", "path"])
def test_emission_adds_to_reflected_light(device_, tracer):
    """A lit surface that also emits is the sum of the two.

    Emission is added at the hit and the surface goes on scattering, so it
    neither replaces the shading nor is attenuated by it.
    """
    emission = (0.2, 0.2, 0.2)
    lit = _center(_render(_sphere_scene(device_, lit=True), tracer))
    lit_and_emissive = _center(
        _render(_sphere_scene(device_, lit=True, emission=emission), tracer)
    )

    assert lit.max() > 0.01
    numpy.testing.assert_allclose(
        lit_and_emissive, lit + numpy.array(emission), atol=1e-5
    )


@pytest.mark.parametrize("tracer", ["preview", "path"])
def test_solid_material_can_emit(device_, tracer):
    """``solid`` is a flat color, and emission adds on top of it."""
    color = (0.2, 0.2, 0.2)
    emission = (0.5, 0.25, 0.125)
    buf = _render(
        _sphere_scene(device_, solid=1.0, color=color, emission=emission), tracer
    )

    numpy.testing.assert_allclose(
        _center(buf), numpy.array(color) + numpy.array(emission), atol=1e-6
    )


@pytest.mark.parametrize("tracer", ["preview", "path"])
def test_outline_material_can_emit(device_, tracer):
    """An outline carries its own emission, so an outline can glow on its own."""
    scene = _sphere_scene(device_)
    scene.geometry[0].outline_material = fresnel.material.Material(
        emission=(1.0, 0.0, 0.0)
    )
    scene.geometry[0].outline_width = 0.2

    buf = _render(scene, tracer, w=48, h=48)

    # the sphere more than fills the view, so the outline is off screen and
    # the middle is the un-emissive material: nothing glows
    numpy.testing.assert_allclose(_center(buf), (0.0, 0.0, 0.0), atol=1e-6)

    # pull the camera back and the rim is there, glowing red
    scene.camera.height = 4
    buf = _render(scene, tracer, w=48, h=48)
    rim = buf[:, :, 0].max()
    assert rim == pytest.approx(1.0, abs=1e-5)
    assert buf[:, :, 1].max() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# what an emitter does to the rest of the scene
# ---------------------------------------------------------------------------


def _two_sphere_scene(device, emission):
    """Build a scene with a white diffuse sphere beside an emissive one."""
    scene = _dark_scene(device)

    lamp = fresnel.geometry.Sphere(scene, position=[[-1.6, 0, 0]], radius=1.0)
    lamp.material = fresnel.material.Material(emission=emission)

    wall = fresnel.geometry.Sphere(scene, position=[[1.6, 0, 0]], radius=1.0)
    wall.material = fresnel.material.Material(color=(0.9, 0.9, 0.9), roughness=1.0)

    # look at the diffuse sphere only: the lamp is off the side of the frame
    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 8), look_at=(1.6, 0, 0), up=(0, 1, 0), height=1.2
    )

    return scene


def test_an_emitter_lights_the_scene(device_):
    """``tracer.Path`` picks up the light an emissive surface casts.

    There is no next event estimation here: a path finds the emitter only by
    hitting it, exactly as it finds a ``light.Light``. That needs no new
    machinery, and it is why a small bright emitter is noisier than a large
    dim one.
    """
    lit = _center(
        _render(_two_sphere_scene(device_, (8.0, 8.0, 8.0)), "path", samples=96)
    )
    unlit = _center(
        _render(_two_sphere_scene(device_, (0.0, 0.0, 0.0)), "path", samples=96)
    )

    # the diffuse sphere does not emit, so without the lamp it is black
    numpy.testing.assert_allclose(unlit, (0.0, 0.0, 0.0), atol=1e-6)
    assert lit.min() > 0.0


def test_preview_shows_no_light_cast_by_an_emitter(device_):
    """``tracer.Preview`` traces no scattered rays, so it cannot show it.

    This is the same limitation that makes the preview miss soft shadows, and
    it is worth pinning down: a preview of an emissive scene shows the emitter
    and not its effect on anything else.
    """
    lit = _center(_render(_two_sphere_scene(device_, (8.0, 8.0, 8.0)), "preview"))

    numpy.testing.assert_allclose(lit, (0.0, 0.0, 0.0), atol=1e-6)


def test_emission_is_absorbed_by_a_material_it_shines_through(device_):
    """Light from an emitter behind a solid is absorbed by that solid.

    ``ior=1`` leaves an interface that neither bends nor reflects, so the only
    thing acting on the light between the emitter and the camera is
    Beer-Lambert absorption over the thickness the path crossed. That makes
    the expected value exact: ``emission * color ** (thickness /
    transmission_distance)``.
    """
    emission = 0.8
    color = 0.5
    transmission_distance = 4.0

    scene = _slab_scene(
        device_,
        emission=(emission,) * 3,
        color=(color,) * 3,
        spec_trans=1.0,
        ior=1.0,
        transmission_distance=transmission_distance,
    )

    measured = _center(_render(scene, "path", samples=64))
    expected = emission * color ** (SLAB_THICKNESS / transmission_distance)

    numpy.testing.assert_allclose(measured, (expected,) * 3, rtol=2e-3)


@pytest.mark.parametrize("tracer", ["preview", "path"])
def test_emission_is_two_sided(device_, tracer):
    """A surface that glows glows on both of its faces.

    A ray through a clear emissive solid crosses the surface twice, once on
    the way in and once on the way out, and picks the emission up each time.
    Both tracers double it: the path tracer emits at the back face after the
    interior has absorbed along the segment, and the preview emits at each
    interface it crosses.
    """
    emission = 0.3
    one_face = _center(
        _render(_sphere_scene(device_, emission=(emission,) * 3), tracer)
    )
    two_faces = _center(
        _render(
            _sphere_scene(
                device_,
                emission=(emission,) * 3,
                # clear all the way through: nothing bends, nothing reflects,
                # and the interior absorbs nothing worth speaking of
                spec_trans=1.0,
                ior=1.0,
                roughness=0.0,
                color=(1.0, 1.0, 1.0),
                transmission_distance=1e6,
            ),
            tracer,
        )
    )

    numpy.testing.assert_allclose(one_face, (emission,) * 3, atol=1e-6)
    numpy.testing.assert_allclose(two_faces, (2.0 * emission,) * 3, atol=1e-5)
