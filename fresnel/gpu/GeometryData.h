// Copyright (c) 2016-2026 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef GEOMETRY_DATA_H_
#define GEOMETRY_DATA_H_

#include "common/ColorMath.h"
#include "common/GeometryMath.h"
#include "common/IntersectCylinder.h"
#include "common/IntersectSphere.h"
#include "common/IntersectTriangle.h"
#include "common/VectorMath.h"
#include "cuda_platform.h"

#include <cfloat>

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#undef DEVICE
#ifdef __CUDACC__
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

namespace fresnel
    {
namespace gpu
    {
//! The primitive types a Geometry may contribute to the Scene
enum GeometryType : unsigned int
    {
    geometry_sphere = 0,
    geometry_cylinder,
    geometry_polygon,
    geometry_mesh,
    geometry_convex_polyhedron
    };

//! Device visible description of one Geometry
/*! Embree calls back into a per-geometry bounds and intersect function, and the geometry object
   itself is the callback's user data. There is no callback to hook here: a megakernel switches on
   \a type instead, so every field a primitive needs has to be reachable from one plain old data
   struct.

    All of the pointers refer to managed allocations owned by the Geometry that filled the struct
   out. Geometry::update() refreshes the struct in the Scene, so a GeometryData is only valid for
   as long as its Geometry is alive and attached.

    Not every field applies to every type. The unused ones are null.
*/
struct GeometryData
    {
    unsigned int type; //!< One of GeometryType
    unsigned int n_primitives; //!< Number of primitives this geometry puts in the BVH
    unsigned int enabled; //!< 0 when the geometry should be skipped during traversal

    // per-object buffers
    const vec3<float>* position; //!< Sphere, cylinder (2 per primitive), mesh and polyhedron
                                 //!< positions
    const vec2<float>* position_2d; //!< Polygon positions
    const float* radius; //!< Sphere and cylinder radii
    const quat<float>* orientation; //!< Mesh and polyhedron orientations
    const float* angle; //!< Polygon orientations
    const RGB<float>* color; //!< Per primitive color (2 per cylinder, 3 per mesh face)

    // polygon shape
    const vec2<float>* polygon_vertices; //!< Polygon vertices, counterclockwise
    unsigned int n_polygon_vertices; //!< Number of polygon vertices
    float rounding_radius; //!< Spheropolygon rounding radius
    float polygon_radius; //!< Bounding radius of the polygon in the xy plane

    // mesh shape
    const vec3<float>* mesh_vertices; //!< Triangle vertices, 3 per face
    unsigned int n_faces; //!< Number of faces in one mesh

    // convex polyhedron shape
    const vec3<float>* plane_origin; //!< Origin of each bounding plane
    const vec3<float>* plane_normal; //!< Outward normal of each bounding plane
    const RGB<float>* plane_color; //!< Color of each bounding plane
    unsigned int n_planes; //!< Number of bounding planes
    float polyhedron_radius; //!< Bounding radius of the polyhedron
    float color_by_face; //!< Mixes the per particle color with the per face color
    };

//! Test if a point is inside a polygon
/*! \param min_d  [out] minimum distance from p to the polygon edge
    \param p Point
    \param verts Polygon vertices
    \param nvert Number of polygon vertices

    \returns true if the point is inside the polygon

    \note \a p is *in the polygon's reference frame!*
*/
DEVICE inline bool
is_inside(float& min_d, const vec2<float>& p, const vec2<float>* verts, unsigned int nvert)
    {
    // code for concave test from: http://alienryderflex.com/polygon/
    min_d = FLT_MAX;

    unsigned int i, j = nvert - 1;
    bool oddNodes = false;

    for (i = 0; i < nvert; i++)
        {
        min_d = fast::min(min_d, point_line_segment_distance(p, verts[i], verts[j]));

        if ((verts[i].y < p.y && verts[j].y >= p.y) || (verts[j].y < p.y && verts[i].y >= p.y))
            {
            if (verts[i].x
                    + (p.y - verts[i].y) / (verts[j].y - verts[i].y) * (verts[j].x - verts[i].x)
                < p.x)
                {
                oddNodes = !oddNodes;
                }
            }
        j = i;
        }

    return oddNodes;
    }

//! Compute the axis aligned bounding box of one primitive
/*! \param geom Geometry the primitive belongs to
    \param prim_id Index of the primitive within the geometry
    \param lower [out] Lower corner of the box
    \param upper [out] Upper corner of the box

    This is the equivalent of the Embree bounds callback in each cpu/Geometry*.cc.
*/
DEVICE inline void geometry_bounds(const GeometryData& geom,
                                   unsigned int prim_id,
                                   vec3<float>& lower,
                                   vec3<float>& upper)
    {
    switch (geom.type)
        {
    case geometry_sphere:
        {
        const vec3<float> p = geom.position[prim_id];
        const float radius = geom.radius[prim_id];
        lower = p - vec3<float>(radius, radius, radius);
        upper = p + vec3<float>(radius, radius, radius);
        break;
        }

    case geometry_cylinder:
        {
        const vec3<float> A = geom.position[prim_id * 2 + 0];
        const vec3<float> B = geom.position[prim_id * 2 + 1];
        const float radius = geom.radius[prim_id];

        lower = vec3<float>(fast::min(A.x - radius, B.x - radius),
                            fast::min(A.y - radius, B.y - radius),
                            fast::min(A.z - radius, B.z - radius));
        upper = vec3<float>(fast::max(A.x + radius, B.x + radius),
                            fast::max(A.y + radius, B.y + radius),
                            fast::max(A.z + radius, B.z + radius));
        break;
        }

    case geometry_polygon:
        {
        const vec2<float> p2 = geom.position_2d[prim_id];
        const float r = geom.polygon_radius;
        lower = vec3<float>(p2.x - r, p2.y - r, -1e-5f);
        upper = vec3<float>(p2.x + r, p2.y + r, 1e-5f);
        break;
        }

    case geometry_mesh:
        {
        const unsigned int i_poly = prim_id / geom.n_faces;
        const unsigned int i_face = prim_id % geom.n_faces;

        const vec3<float> p3 = geom.position[i_poly];
        const quat<float> q_world = geom.orientation[i_poly];

        const vec3<float> v0 = rotate(q_world, geom.mesh_vertices[i_face * 3 + 0]) + p3;
        const vec3<float> v1 = rotate(q_world, geom.mesh_vertices[i_face * 3 + 1]) + p3;
        const vec3<float> v2 = rotate(q_world, geom.mesh_vertices[i_face * 3 + 2]) + p3;

        lower = vec3<float>(fast::min(v0.x, fast::min(v1.x, v2.x)),
                            fast::min(v0.y, fast::min(v1.y, v2.y)),
                            fast::min(v0.z, fast::min(v1.z, v2.z)));
        upper = vec3<float>(fast::max(v0.x, fast::max(v1.x, v2.x)),
                            fast::max(v0.y, fast::max(v1.y, v2.y)),
                            fast::max(v0.z, fast::max(v1.z, v2.z)));
        break;
        }

    case geometry_convex_polyhedron:
    default:
        {
        const vec3<float> p = geom.position[prim_id];
        const float r = geom.polyhedron_radius;
        lower = p - vec3<float>(r, r, r);
        upper = p + vec3<float>(r, r, r);
        break;
        }
        }
    }

//! Intersect a ray with one primitive
/*! \param geom Geometry the primitive belongs to
    \param prim_id Index of the primitive within the geometry
    \param org Ray origin
    \param dir Ray direction (normalized)
    \param t_near Near clipping distance
    \param t_far Far clipping distance, the distance to the closest hit found so far
    \param hit [out] Hit record, filled out when this returns true

    \returns true when the primitive is hit in (t_near, t_far)

    This is the equivalent of the Embree intersect callback in each cpu/Geometry*.cc. \a hit
   receives the same attributes Embree carries out of an intersection: the distance, the distance to
   the nearest edge, the shading color, and the *geometric* normal. The normal points out of the
   primitive whichever side was struck; flipping it to the viewing side is the tracer's job.
*/
DEVICE inline bool geometry_intersect(const GeometryData& geom,
                                      unsigned int prim_id,
                                      const vec3<float>& org,
                                      const vec3<float>& dir,
                                      float t_near,
                                      float t_far,
                                      HitInfo& hit)
    {
    switch (geom.type)
        {
    case geometry_sphere:
        {
        const vec3<float> position = geom.position[prim_id];
        const float radius = geom.radius[prim_id];

        float t = 0, d = 0;
        vec3<float> N;
        if (!intersect_ray_sphere(t, d, N, org, dir, position, radius))
            return false;

        if (!((t_near < t) && (t < t_far)))
            return false;

        hit.t = t;
        hit.d = d;
        hit.Ng = org + t * dir - position;
        hit.shading_color = geom.color[prim_id];
        return true;
        }

    case geometry_cylinder:
        {
        const vec3<float> A = geom.position[prim_id * 2 + 0];
        const vec3<float> B = geom.position[prim_id * 2 + 1];
        const float radius = geom.radius[prim_id];

        float t = HUGE_VALF, d = HUGE_VALF;
        vec3<float> N;
        unsigned int color_index;
        if (!intersect_ray_spherocylinder(t, d, N, color_index, org, dir, A, B, radius))
            return false;

        if (!((t_near < t) && (t < t_far)))
            return false;

        hit.t = t;
        hit.d = d;
        hit.Ng = N;
        hit.shading_color = geom.color[prim_id * 2 + color_index];
        return true;
        }

    case geometry_polygon:
        {
        const vec2<float> p2 = geom.position_2d[prim_id];
        const vec3<float> pos_world(p2.x, p2.y, 0.0f);
        const float angle = geom.angle[prim_id];
        const quat<float> q_world = quat<float>::fromAxisAngle(vec3<float>(0, 0, 1), angle);

        // transform the ray into the primitive coordinate system
        const vec3<float> ray_dir_local = rotate(conj(q_world), dir);
        const vec3<float> ray_org_local = rotate(conj(q_world), org - pos_world);

        // find the point where the ray intersects the plane of the polygon
        const vec3<float> n = vec3<float>(0, 0, 1);
        const vec3<float> p = vec3<float>(0, 0, 0);

        const float d_plane = -dot(n, p);
        const float denom = dot(n, ray_dir_local);

        // if the ray is parallel to the plane, there is no intersection
        if (fabsf(denom) < 1e-5f)
            return false;

        const float t_hit = -(d_plane + dot(n, ray_org_local)) / denom;

        // see if the intersection point is inside the polygon
        const vec3<float> r_hit = ray_org_local + t_hit * ray_dir_local;
        const vec2<float> r_hit_2d(r_hit.x, r_hit.y);
        float d_edge;

        const bool inside
            = is_inside(d_edge, r_hit_2d, geom.polygon_vertices, geom.n_polygon_vertices);

        // spheropolygon (equivalent to sharp polygon when rounding radius is 0)
        // make distance signed (negative is inside)
        if (inside)
            d_edge = -d_edge;

        // exit if outside
        if (d_edge > geom.rounding_radius)
            return false;

        if (!((t_near < t_hit) && (t_hit < t_far)))
            return false;

        hit.t = t_hit;
        hit.d = geom.rounding_radius - d_edge;

        // Polygons are double sided, but report the geometric normal of the plane rather than
        // one flipped toward the ray. The tracers flip it to the viewing side for shading and
        // use the unflipped normal to tell which face was hit.
        hit.Ng = rotate(q_world, n);
        hit.shading_color = geom.color[prim_id];
        return true;
        }

    case geometry_mesh:
        {
        const unsigned int i_poly = prim_id / geom.n_faces;
        const unsigned int i_face = prim_id % geom.n_faces;

        const vec3<float> p3 = geom.position[i_poly];
        const quat<float> q_world = geom.orientation[i_poly];

        // transform the ray into the primitive coordinate system
        const vec3<float> ray_dir_local = rotate(conj(q_world), dir);
        const vec3<float> ray_org_local = rotate(conj(q_world), org - p3);

        const vec3<float> v0 = geom.mesh_vertices[i_face * 3 + 0];
        const vec3<float> v1 = geom.mesh_vertices[i_face * 3 + 1];
        const vec3<float> v2 = geom.mesh_vertices[i_face * 3 + 2];
        float u, v, w, t, d;
        vec3<float> n;

        // double-sided triangle test. intersect_ray_triangle() culls back faces, so a miss is
        // retried against the reversed winding to catch hits on the far side of the triangle.
        if (!intersect_ray_triangle(u,
                                    v,
                                    w,
                                    t,
                                    d,
                                    n,
                                    ray_org_local,
                                    ray_org_local + ray_dir_local,
                                    v0,
                                    v1,
                                    v2))
            {
            if (!intersect_ray_triangle(v,
                                        u,
                                        w,
                                        t,
                                        d,
                                        n,
                                        ray_org_local,
                                        ray_org_local + ray_dir_local,
                                        v1,
                                        v0,
                                        v2))
                return false;

            // The retry reversed the winding, so it returns a normal pointing along the ray.
            // Flip it back to recover the triangle's geometric normal: the tracers rely on the
            // reported normal identifying which face was hit.
            n = -n;
            }

        if (!((t_near < t) && (t < t_far)))
            return false;

        hit.t = t;
        hit.d = d;
        hit.Ng = rotate(q_world, n);
        hit.shading_color = geom.color[i_face * 3 + 0] * u + geom.color[i_face * 3 + 1] * v
                            + geom.color[i_face * 3 + 2] * w;
        return true;
        }

    case geometry_convex_polyhedron:
    default:
        {
        // adapted from OptiX quick start tutorial and Embree user_geometry tutorial files
        const unsigned int n_planes = geom.n_planes;
        float t0 = -FLT_MAX;
        float t1 = FLT_MAX;

        const vec3<float> pos_world = geom.position[prim_id];
        const quat<float> q_world = geom.orientation[prim_id];

        // transform the ray into the primitive coordinate system
        const vec3<float> ray_dir_local = rotate(conj(q_world), dir);
        const vec3<float> ray_org_local = rotate(conj(q_world), org - pos_world);

        vec3<float> t0_n_local(0, 0, 0), t0_p_local(0, 0, 0);
        vec3<float> t1_n_local(0, 0, 0), t1_p_local(0, 0, 0);
        unsigned int t0_plane_hit = 0, t1_plane_hit = 0;
        for (unsigned int i = 0; i < n_planes && t0 <= t1; ++i)
            {
            const vec3<float> n = geom.plane_normal[i];
            const vec3<float> p = geom.plane_origin[i];

            const float d = -dot(n, p);
            const float denom = dot(n, ray_dir_local);
            const float t = -(d + dot(n, ray_org_local)) / denom;

            // if the ray is parallel to the plane, there is no intersection when the ray is
            // outside the shape
            if (fabsf(denom) < 1e-5f)
                {
                if (dot(ray_org_local - p, n) > 0)
                    return false;
                }
            else if (denom < 0)
                {
                // find the last plane this ray enters
                if (t > t0)
                    {
                    t0 = t;
                    t0_n_local = n;
                    t0_p_local = p;
                    t0_plane_hit = i;
                    }
                }
            else
                {
                // find the first plane this ray exits
                if (t < t1)
                    {
                    t1 = t;
                    t1_n_local = n;
                    t1_p_local = p;
                    t1_plane_hit = i;
                    }
                }
            }

        // if the ray enters after it exits, it missed the polyhedron
        if (t0 > t1)
            return false;

        // otherwise, it hit: fill out the hit structure and track the plane that was hit
        // t_far narrows as the entry plane is accepted, exactly as ray.tfar does under Embree,
        // so the exit plane is only reported when the entry plane was missed
        float t_limit = t_far;
        float t_hit = 0;
        bool found = false;
        vec3<float> n_hit(0, 0, 0), p_hit(0, 0, 0);

        // if the t0 is in (tnear,tfar), we hit the entry plane
        if ((t_near < t0) && (t0 < t_limit))
            {
            t_hit = t_limit = t0;
            hit.Ng = rotate(q_world, t0_n_local);
            n_hit = t0_n_local;
            p_hit = t0_p_local;
            hit.shading_color
                = lerp(geom.color_by_face, geom.color[prim_id], geom.plane_color[t0_plane_hit]);
            found = true;
            }
        // if t1 is in (tnear,tfar), we hit the exit plane
        if ((t_near < t1) && (t1 < t_limit))
            {
            t_hit = t_limit = t1;
            hit.Ng = rotate(q_world, t1_n_local);
            n_hit = t1_n_local;
            p_hit = t1_p_local;
            hit.shading_color
                = lerp(geom.color_by_face, geom.color[prim_id], geom.plane_color[t1_plane_hit]);
            found = true;
            }

        if (!found)
            return false;

        // determine distance from the hit point to the nearest edge
        float min_d = FLT_MAX;
        const vec3<float> r_hit = ray_org_local + t_hit * ray_dir_local;

        // edges come from intersections of planes
        // loop over all planes and find the intersection with the hit plane
        for (unsigned int i = 0; i < n_planes; ++i)
            {
            const vec3<float> n = geom.plane_normal[i];
            const vec3<float> p = geom.plane_origin[i];

            // ********
            // find the line of intersection between the two planes
            // adapted from: http://geomalgorithms.com/a05-_intersect-1.html

            // direction of the line
            vec3<float> u = cross(n, n_hit);

            // if the planes are not coplanar
            if (fabsf(dot(u, u)) >= 1e-5f)
                {
                int maxc; // max coordinate
                if (fabsf(u.x) > fabsf(u.y))
                    {
                    if (fabsf(u.x) > fabsf(u.z))
                        maxc = 1;
                    else
                        maxc = 3;
                    }
                else
                    {
                    if (fabsf(u.y) > fabsf(u.z))
                        maxc = 2;
                    else
                        maxc = 3;
                    }

                // a point on the line
                vec3<float> x0(0, 0, 0);
                const float d1 = -dot(n, p);
                const float d2 = -dot(n_hit, p_hit);

                // solve the problem in different ways based on which direction is maximum
                switch (maxc)
                    {
                case 1: // intersect with x=0
                    x0.x = 0;
                    x0.y = (d2 * n.z - d1 * n_hit.z) / u.x;
                    x0.z = (d1 * n_hit.y - d2 * n.y) / u.x;
                    break;
                case 2: // intersect with y=0
                    x0.x = (d1 * n_hit.z - d2 * n.z) / u.y;
                    x0.y = 0;
                    x0.z = (d2 * n.x - d1 * n_hit.x) / u.y;
                    break;
                case 3: // intersect with z=0
                    x0.x = (d2 * n.y - d1 * n_hit.y) / u.z;
                    x0.y = (d1 * n_hit.x - d2 * n.x) / u.z;
                    x0.z = 0;
                    }

                // we want the distance in the view plane for consistent line edge widths
                // project the line x0 + t*u into the plane perpendicular to the view direction
                // passing through r_hit
                const vec3<float> view = -ray_dir_local / sqrtf(dot(ray_dir_local, ray_dir_local));
                u = u - dot(u, view) * view;
                const vec3<float> w = x0 - r_hit;
                const vec3<float> w_perp = w - dot(w, view) * view;
                x0 = r_hit + w_perp;

                // ********
                // find the distance from the hit point to the line
                // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
                const vec3<float> v = cross(u, x0 - r_hit);
                const float dsq = dot(v, v) / dot(u, u);
                const float d = sqrtf(dsq);
                if (d < min_d)
                    min_d = d;
                }
            }

        hit.t = t_hit;
        hit.d = min_d;
        return true;
        }
        }
    }

    } // namespace gpu
    } // namespace fresnel

#undef DEVICE

#endif
