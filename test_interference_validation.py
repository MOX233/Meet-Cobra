"""Independent checks for beam directions, normalization and conditional overlap."""
import types
import unittest
import numpy as np
import torch
from experiment.interference_validation import (rb_owners, expected_interference,
    explicit_interference, directional_gains)


class InterferenceValidationTests(unittest.TestCase):
    def test_complete_dft_mean(self):
        rng = np.random.default_rng(21)
        nr, nt = 4, 8
        h = rng.normal(size=(nr,nt)) + 1j*rng.normal(size=(nr,nt))
        rx = np.exp(-2j*np.pi*np.outer(np.arange(nr),np.arange(nr))/nr)/np.sqrt(nr)
        tx = np.exp(-2j*np.pi*np.outer(np.arange(nt),np.arange(nt))/nt)/np.sqrt(nt)
        np.testing.assert_allclose(np.mean(np.abs(rx.conj().T @ h @ tx)**2),
                                   np.sum(np.abs(h)**2)/(nr*nt), rtol=1e-14)

    def test_actual_cross_beams_and_normalization(self):
        rng = np.random.default_rng(4)
        slots, vehicles, nr, nb, nt = 3, 5, 4, 4, 8
        h = rng.normal(size=(slots,vehicles,nr,nb,nt)) + 1j*rng.normal(size=(slots,vehicles,nr,nb,nt))
        rx = np.exp(-2j*np.pi*np.outer(np.arange(nr),np.arange(nr))/nr)
        tx = np.exp(-2j*np.pi*np.outer(np.arange(nt),np.arange(nt))/nt)
        pairs = rng.integers(0,nr*nt,(slots,vehicles,nb))
        bs = np.array([1,2,3,4,0])
        physical = types.SimpleNamespace(h=torch.tensor(h),rx=torch.tensor(rx),tx=torch.tensor(tx),
            device=torch.device('cpu'), args=types.SimpleNamespace(M_r=nr,M_t=nt))
        cross, mean, own = directional_gains(physical,pairs,bs)
        expected = np.zeros_like(cross)
        for s in range(slots):
            for v in range(vehicles):
                r = pairs[s,v,max(bs[v]-1,0)] % nr
                for w in range(vehicles):
                    b = max(bs[w]-1,0)
                    t = pairs[s,w,b] // nr
                    expected[s,v,w] = abs(rx[:,r].conj() @ h[s,v,:,b,:] @ tx[:,t])**2/(nr*nt)
        np.testing.assert_allclose(cross,expected,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(own,np.diagonal(expected,axis1=1,axis2=2))
        np.testing.assert_allclose(mean,np.mean(abs(h)**2,axis=(2,4)))

    def test_owners_and_no_intracell_overlap(self):
        bs, counts = np.array([0,1,1,2,4]),np.array([133,2,3,4,0])
        owners = rb_owners(bs,counts,8,np.random.default_rng(1))
        for v in range(1,len(bs)):
            self.assertEqual(np.sum(owners==v),counts[v])
        self.assertFalse(np.any(owners==0))
        self.assertTrue(np.all(owners[2:] == -1))
        with self.assertRaises(AssertionError):
            rb_owners(bs,counts,4,np.random.default_rng(1))

    def test_no_other_cell_interference(self):
        bs, counts = np.array([1,1,0]),np.array([2,2,3])
        gain = np.ones((3,3))
        owners = rb_owners(bs,counts,4,np.random.default_rng(1))
        e = expected_interference(gain,bs,counts,4,1)
        actual,mask = explicit_interference(gain,bs,owners,1)
        np.testing.assert_array_equal(e,0)
        np.testing.assert_array_equal(actual,0)
        np.testing.assert_array_equal(mask.sum(1),[2,2,0])

    def test_exhaustive_two_cell_expected_interference(self):
        # K=2, one occupied RB per BS: exhaust all independent permutations.
        bs, counts = np.array([1,2]),np.array([1,1])
        gain = np.array([[99.,3.],[5.,99.]])
        realized = []
        for a in range(2):
            for b in range(2):
                owners = np.full((4,2),-1,dtype=int)
                owners[0,a],owners[1,b]=0,1
                inter,mask = explicit_interference(gain,bs,owners,1)
                realized.append((inter*mask).sum(1))
        np.testing.assert_allclose(np.mean(realized,axis=0),expected_interference(gain,bs,counts,2,1))

    def test_random_overlap_and_directional_expectation(self):
        bs,counts=np.array([1,1,2,2,3]),np.array([2,3,1,3,2])
        rng=np.random.default_rng(33)
        gain=rng.uniform(.1,2,(5,5))
        observed=[]
        overlaps=[]
        for _ in range(10000):
            owners=rb_owners(bs,counts,8,rng)
            inter,mask=explicit_interference(gain,bs,owners,.2)
            observed.append((inter*mask).sum(1)/counts)
            overlaps.append(np.sum((owners[0]>=0)&(owners[1]>=0)))
        observed=np.asarray(observed)
        target=expected_interference(gain,bs,counts,8,.2)
        self.assertTrue(np.all(abs(observed.mean(0)-target) < 5*observed.std(0)/np.sqrt(len(observed))))
        self.assertLess(abs(np.mean(overlaps)-5*4/8),5*np.std(overlaps)/np.sqrt(len(overlaps)))


if __name__ == '__main__':
    unittest.main()
