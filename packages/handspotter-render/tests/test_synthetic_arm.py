"""Check the joint and mesh invariants of the synthetic arm generator."""
import unittest

import numpy as np

from replay_surface import wrist_boundary
from synthetic_arm import Body, arm_clearance_samples, arm_surface, hemisphere_arm, pose_hand, sample_arm, wide_reach_target


def angle(a, b):
    dot = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return np.rad2deg(np.arccos(np.clip(dot, -1, 1)))


class SyntheticArmTests(unittest.TestCase):
    def test_reach_keeps_segment_lengths_and_joint_limits(self):
        rng = np.random.default_rng(501)
        for _ in range(140):
            body = Body.sample(rng)
            for side in (0, 1):
                arm = sample_arm(rng, body, side)
                forearm = arm.wrist-arm.elbow
                self.assertAlmostEqual(np.linalg.norm(forearm), body.forearm_length)
                self.assertAlmostEqual(np.linalg.norm(arm.elbow-arm.shoulder), body.upper_length)
                elbow_angle = angle(forearm, arm.shoulder-arm.elbow)
                self.assertGreaterEqual(elbow_angle, 55.-1e-9)
                self.assertLessEqual(elbow_angle, 155.+1e-9)
                self.assertLessEqual(angle(forearm, arm.hand_rotation[:, 1]), 43.+1e-9)
                self.assertAlmostEqual(np.linalg.det(arm.hand_rotation), 1.)

    def test_hand_follows_wrist_without_deforming_the_fingers(self):
        rng = np.random.default_rng(80)
        joints = rng.normal(0., .03, (21, 3))
        joints[0] = [0., 0., 0.]
        joints[[5,9,13,17]] = [[.03,.068,.004],[.01,.072,0.],
                               [-.01,.066,.001],[-.03,.056,.007]]
        scale = .93
        for side in (0, 1):
            arm = sample_arm(rng, Body.sample(rng), side)
            vertices, moved = pose_hand({"vertices": joints, "joints": joints}, arm, side, scale)
            np.testing.assert_allclose(vertices, moved)
            np.testing.assert_allclose(moved[0], arm.wrist)
            np.testing.assert_allclose(np.linalg.norm(np.diff(moved, axis=0), axis=1),
                                       np.linalg.norm(np.diff(joints, axis=0), axis=1)*scale)
            palm = moved[[5,9,13,17]].mean(axis=0)-arm.wrist
            np.testing.assert_allclose(palm/np.linalg.norm(palm),arm.hand_rotation[:, 1])

    def test_wide_reach_adds_near_far_and_high_wrists(self):
        rng = np.random.default_rng(931)
        wrists = []
        for _ in range(150):
            body = Body.sample(rng)
            for side in (0,1):
                target = wide_reach_target(rng,side)
                arm = sample_arm(rng,body,side,target=target,wide=True)
                forearm = arm.wrist-arm.elbow
                self.assertAlmostEqual(np.linalg.norm(forearm),body.forearm_length)
                self.assertAlmostEqual(np.linalg.norm(arm.elbow-arm.shoulder),body.upper_length)
                self.assertGreaterEqual(angle(forearm,arm.shoulder-arm.elbow),40.-1e-9)
                self.assertLessEqual(angle(forearm,arm.shoulder-arm.elbow),166.+1e-9)
                wrists.append(arm.wrist)
        wrists = np.asarray(wrists)
        self.assertGreater(np.quantile(wrists[:,2],.9),-.25)
        self.assertLess(np.quantile(wrists[:,2],.1),-.43)
        self.assertGreater(np.quantile(wrists[:,1],.9),0.)

    def test_hemisphere_reaches_cover_all_sectors_and_separate_arms(self):
        rng = np.random.default_rng(812)
        sectors = set()
        for _ in range(160):
            body = Body.sample(rng)
            left = hemisphere_arm(rng,body,0)
            right = hemisphere_arm(rng,body,1,left)
            for arm in (left,right):
                span = np.linalg.norm(arm.wrist-arm.elbow)
                self.assertAlmostEqual(span, body.forearm_length)
                self.assertAlmostEqual(np.linalg.norm(arm.elbow-arm.shoulder), body.upper_length)
                self.assertLess(arm.wrist[2], body.shoulder_z-.14)
                offset = arm.wrist - [0.,body.shoulder_y-.07,body.shoulder_z-.14]
                sectors.add(int(np.floor(6*np.arctan2(offset[1],offset[0])/np.pi)))
            a, ar = arm_clearance_samples(left)
            b, br = arm_clearance_samples(right)
            self.assertTrue(np.all(np.linalg.norm(a[:,None]-b,axis=-1) >= ar[:,None]+br))
        self.assertEqual(sectors, set(range(-6,6)))

    def test_arm_surface_keeps_the_wrist_rim_and_joins_it_to_the_shoulder(self):
        rest = np.array([[.025,0.,0.],[0.,0.,.018],[-.025,0.,0.],
                         [0.,0.,-.018],[0.,.052,0.]])
        base = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        rng = np.random.default_rng(11)
        for side in (0, 1):
            arm = sample_arm(rng, Body.sample(rng), side)
            sign = -1 if side == 0 else 1
            faces = base[:, ::-1] if side == 0 else base
            original = (rest*[sign,1,1])@arm.hand_rotation.T+arm.wrist
            rim, edges = wrist_boundary(faces)
            world, material, triangles = arm_surface(original, rest, faces, (rim,edges), arm)
            np.testing.assert_array_equal(world[:len(rest)],original)
            np.testing.assert_array_equal(material[:len(rest)],rest)
            np.testing.assert_array_equal(triangles[:len(base)],faces)
            opening, _ = wrist_boundary(triangles)
            np.testing.assert_array_equal(opening,np.arange(len(world)-len(rim),len(world)))
            np.testing.assert_allclose(world[opening].mean(axis=0),arm.shoulder,atol=1e-12)
            points=world[triangles]
            area=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)
            self.assertTrue(np.all(area>0.))

    def test_flat_wrist_does_not_flatten_the_forearm(self):
        phase = np.arange(16)*2*np.pi/16
        rim_points = np.column_stack((.030*np.cos(phase), np.zeros_like(phase), .021*np.sin(phase)))
        hand = np.vstack((rim_points, [0., .060, 0.]))
        faces = np.array([(i,(i+1)%16,16) for i in range(16)])
        boundary = wrist_boundary(faces)
        rng = np.random.default_rng(9)
        arm = sample_arm(rng, Body.sample(rng), 1)

        rings = []
        for rest in (hand, hand*[1,1,.40]):
            vertices = rest@arm.hand_rotation.T+arm.wrist
            world, _, _ = arm_surface(vertices, rest, faces, boundary, arm)
            np.testing.assert_array_equal(world[:len(rest)], vertices)
            rings.append(world[len(rest):].reshape(-1,16,3))

        self.assertGreater(np.max(np.abs(rings[0][0]-rings[1][0])), .005)
        np.testing.assert_allclose(rings[0][7:], rings[1][7:], atol=1e-12)
        centered = rings[1][8]-rings[1][8].mean(axis=0)
        major, minor = np.linalg.svd(centered, compute_uv=False)[:2]
        self.assertGreater(minor/major, .75)


if __name__ == "__main__":
    unittest.main()
