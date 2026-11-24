"""Check that torso sockets close and attach to the oriented arm boundaries."""

import unittest

import numpy as np

from replay_surface import wrist_boundary
from synthetic_arm import Body, arm_surface, sample_arm
from synthetic_torso import SHOULDER_INSET, torso_surface


class SyntheticTorsoTests(unittest.TestCase):
    def test_torso_with_zero_one_or_two_arms_forms_one_closed_surface(self):
        phase = np.arange(16)*2*np.pi/16
        rim = np.column_stack((.030*np.cos(phase),np.zeros_like(phase),.021*np.sin(phase)))
        rest = np.vstack((rim,[0.,.055,0.]))
        hand_faces = np.array([(i,(i+1)%16,16) for i in range(16)])
        rng = np.random.default_rng(151)
        for active in ((),(0,),(1,),(0,1)):
            with self.subTest(sides=active):
                body = Body.sample(rng)
                arms,openings = [],{}
                for side in active:
                    arm = sample_arm(rng,body,side)
                    faces = hand_faces[:,::-1] if side==0 else hand_faces
                    indices,edges = wrist_boundary(faces)
                    hand = (rest*[arm.handed,1,1])@arm.hand_rotation.T+arm.wrist
                    vertices,_,mesh = arm_surface(hand,rest,faces,(indices,edges),arm,shoulder_inset=SHOULDER_INSET)
                    openings[side] = vertices[-len(indices):],edges
                    arms.append((vertices,mesh))
                positions,faces,seams = torso_surface(body,openings)
                for side in active:
                    np.testing.assert_array_equal(positions[seams[side]],openings[side][0])

                for vertices,mesh in arms:
                    faces = np.vstack((faces, mesh+len(positions)))
                    positions = np.vstack((positions, vertices))
                welded,mapping = np.unique(positions,axis=0,return_inverse=True)
                faces = mapping[faces]
                triangles = welded[faces]
                normal = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                self.assertTrue(np.all(np.linalg.norm(normal,axis=1)>1e-9))
                directed = np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
                edges,counts = np.unique(np.sort(directed,axis=1),axis=0,return_counts=True)
                np.testing.assert_array_equal(counts,np.full(len(edges),2))
                np.testing.assert_array_equal(np.unique(directed,axis=0,return_counts=True)[1],np.ones(len(directed)))
                self.assertEqual(len(np.unique(faces))-len(edges)+len(faces),2)


if __name__=="__main__":
    unittest.main()
