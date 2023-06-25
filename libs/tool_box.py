# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:23:53 2023

@author: sadifenchen
"""

### This RealNVP model to compute the mapping for free energy binding
#   Success is the application of loss function to the model
### Problem is calculation for larger than 1-2 Angstroms distances
from typing import Sequence, Optional
import numpy as np
import jax 
import jax.numpy as jnp
from flax import linen as nn 
from flax.training import train_state 
from flax.linen.initializers import zeros as nn_zeros
import optax 
import pymbar 
import sys 
import jax_amber
import pickle

from flax.linen.initializers import lecun_normal


default_kernel_init = lecun_normal()
RT = jnp.float32(8.3144621E-3 * 300.0) 
beta = jnp.float32(1.0)/RT 
nm2ang = jnp.float32(10.0) #conversion nanometers -> angstroms
ang2nm = jnp.float32(0.1) #conversion angstroms -> nanometers


def checkpoint_save (fname, ckpt): 
    ### Save 'ckpt' to file. This saves a state of the model.
    with open(fname, 'wb') as fp:
        pickle.dump (ckpt, fp)

def checkpoint_load (fname): 
    ### Load the file.
    with open (fname, 'rb') as fp:
        return pickle.load (fp)

class AfflineCoupling(nn.Module):
    ### RealNVP model
    #   focuses on forward mapping
    ###
    input_size: int #the size of the input data
    i_dim: int #the dimension to be split
    hidden_layers: int #NN's hidden layers
    hidden_dim : int #Neuron in each hidden layers
    fixed_atoms: Sequence[int] # indices of fixed atoms

    @nn.compact
    def __call__ (self, inputs, reverse=False):

        fixed_mask = jnp.ones ((self.input_size), dtype=jnp.int32).reshape(-1,3)
        fixed_mask = fixed_mask.at[:,self.i_dim].set(0)
        moved_mask = jnp.int32(1) - fixed_mask
        moved_mask = moved_mask.at[self.fixed_atoms,self.i_dim].set(0)
        moved_mask = moved_mask.reshape (1,-1)
        fixed_mask = fixed_mask.reshape (1,-1)
        y = inputs*fixed_mask
        
        for _ in range (self.hidden_layers):
            y = nn.relu (nn.Dense (features=self.hidden_dim, kernel_init=default_kernel_init) (y))
    
        log_scale = nn.Dense (features=self.input_size, kernel_init=nn_zeros) (y)
        shift     = nn.Dense (features=self.input_size, kernel_init=nn_zeros) (y)
        shift     = shift*moved_mask 
        log_scale = log_scale*moved_mask
        
        if reverse:
            log_scale = -log_scale
            outputs = (inputs-shift)*jnp.exp(log_scale)
        else:
            outputs = inputs*jnp.exp(log_scale) + shift
      
        return outputs, log_scale 

class realNVP3 (nn.Module):
    ### RealNVP model
    #   focuses on forward mapping
    ###
    input_size: int
    hidden_layers: int
    hidden_dim : int 
    fixed_atoms: Sequence[int]
    
    def setup (self):
        ### initialized instaces of AfflineCoupling
        #   with different 'i_dim'
        ###
        self.af_x = AfflineCoupling (self.input_size, i_dim=0, 
                                     hidden_layers=self.hidden_layers, 
                                     hidden_dim=self.hidden_dim,
                                     fixed_atoms=self.fixed_atoms)
        self.af_y = AfflineCoupling (self.input_size, i_dim=1, 
                                     hidden_layers=self.hidden_layers, 
                                     hidden_dim=self.hidden_dim,
                                     fixed_atoms=self.fixed_atoms)
        self.af_z = AfflineCoupling (self.input_size, i_dim=2, 
                                     hidden_layers=self.hidden_layers, 
                                     hidden_dim=self.hidden_dim,
                                     fixed_atoms=self.fixed_atoms)

    @nn.compact
    def __call__ (self, inputs, reverse=False):
        n_conf, n_atoms, n_dim = inputs.shape 
        
        # reshape the input to 2D tensor for processing
        outputs = inputs.reshape (n_conf, -1)

        if reverse:
            # Reverse operation 
            outputs, log_J_z = self.af_z (outputs, reverse)
            outputs, log_J_y = self.af_y (outputs, reverse)
            outputs, log_J_x = self.af_x (outputs, reverse)
        else:
            # Forward operation 
            outputs, log_J_x = self.af_x (outputs)
            outputs, log_J_y = self.af_y (outputs)
            outputs, log_J_z = self.af_z (outputs)

        # reshape the output tensor back to the original shape
        # and sum the logs of the jacobians
        return outputs.reshape(n_conf, n_atoms, n_dim), \
                (log_J_x + log_J_y + log_J_z).sum(axis=-1)

def get_energy_values (x, ener_funs, R0):
    
    ### Compute energy values using different potential functions.
    #   
    #   x: coordinate array of shape (n_frames, n_atoms, 3)
    #   ener_funs: tuple of 3 functions for bond, harmonic and quartic potentials
    #   R0: reference distance used in harmonic potential function
    #
    ### Returns Tuple containing energy values from each potential function

    ener_nHO_fun, ener_wHO_fun, ener_bond_fun = ener_funs 
    
    # Compute energy from bond potential function
    enr_bnd = jax.vmap(ener_bond_fun) (x)
    
    # Compute energy from harmonic potential function
    enr_nHO = jax.vmap(ener_nHO_fun) (x)
    
    # Compute energy from quartic potential function
    enr_wHO = jax.vmap(ener_wHO_fun, in_axes=(0,None)) (x, R0)
    
    # Return the computed energy values
    return enr_bnd, enr_nHO, enr_wHO

def print_progress (state, inputs, ener_funs, ener_ref0, fixed_iatom, fixed_R0, fout):
    ### Print progress of the model to check the correctness
    #   Returns log written to file in 'fout' variable
    ###
    x_A, x_B = inputs
    R0_A, R0_B, dE0 = fixed_R0
    (enr_nHO_A0, enr_nHO_B0), (enr_wHO_A0, enr_wHO_B0,_,_) = ener_ref0
    
    # Forward mapping
    m_B, log_J_F = state.apply_fn ({'params':state.params}, x_A)
    # Backward mapping
    m_A, log_J_R = state.apply_fn ({'params':state.params}, x_B, reverse=True)
    
    # Compute energy values
    enr_bond_A, enr_nHO_A, enr_wHO_A = get_energy_values (m_A, ener_funs, R0_A)
    enr_bond_B, enr_nHO_B, enr_wHO_B = get_energy_values (m_B, ener_funs, R0_B)
    
    # Calculate diffrence between the forward and backward
    dU_F = enr_wHO_B - enr_wHO_A0
    dU_R = enr_wHO_A - enr_wHO_B0 
    phi_F = beta*dU_F - log_J_F
    phi_R = beta*dU_R - log_J_R 

    # Calculate average z coordinate
    Z_A = m_A[:,fixed_iatom,2].mean()
    Z_B = m_B[:,fixed_iatom,2].mean()

    # Print progress data
    print ('R_A R_B dE          {:12.6f} {:12.6f} {:8.4f}'.format (R0_A[-1], R0_B[-1], dE0), file=fout)
    print ('Fixed_Z             {:12.6f} {:12.6f}'.format(Z_A, Z_B), file=fout)
    print ('<-log_J>(kJ/mol)    {:12.6f} {:12.6f}'.format(RT*(-log_J_F-log_J_R).mean(), -RT*log_J_F.mean()), file=fout)
    print (' <U_bond>(kJ/mol)   {:12.6f} {:12.6f}'.format (enr_bond_A.mean(), enr_bond_B.mean()), file=fout)
    print (' <U_wHO>(kJ/mol)    {:12.6f} {:12.6f}'.format (enr_wHO_A.mean(), enr_wHO_B.mean()), file=fout)
    print ('<dU_wHO>(kJ/mol)    {:12.6f} {:12.6f}'.format (dU_F.mean(), dU_R.mean()), file=fout)
    print ('<phi_wHO>(kJ/mol)   {:12.6f} {:12.6f}'.format (RT*phi_F.mean(), RT*phi_R.mean()), file=fout) 

    # Calculate free energy difference using BAR
    f_BAR_wHO = pymbar.bar (phi_F, phi_R,
                        relative_tolerance=1.0e-5,
                        verbose=False,
                        compute_uncertainty=False)
    print (f_BAR_wHO)
    print ('LBAR(kJ/mol)        {:12.6f}'.format ( RT*f_BAR_wHO['Delta_f']), file=fout)

def loss_value (ener_wHO_fn, ener_bond_fn, enr0_wHO, m_B, log_J_F, m_A, log_J_R, fixed_R0):
    # Extract fixed energy values
    enr_wHO_A0, enr_wHO_B0, enr_bnd_A0, enr_bnd_B0 = enr0_wHO
    R0_A, R0_B, dE0 = fixed_R0

    # Calculate current energy values
    enr_A = jax.vmap(ener_wHO_fn, in_axes=(0,None)) (m_A, R0_A)
    enr_B = jax.vmap(ener_wHO_fn, in_axes=(0,None)) (m_B, R0_B)
    enr_bnd_A = jax.vmap(ener_bond_fn) (m_A)
    enr_bnd_B = jax.vmap(ener_bond_fn) (m_B)

    # Calculate losses for forward and backward directions
    loss_F = beta*(enr_B - enr_wHO_A0) - log_J_F 
    loss_R = beta*(enr_A - enr_wHO_B0) - log_J_R

    # Calculate differences in bond energies
    diff_bnd_A = beta*(enr_bnd_A.mean() - enr_bnd_A0)
    diff_bnd_B = beta*(enr_bnd_B.mean() - enr_bnd_B0)
    diff_bnd_A2 = diff_bnd_A**2
    diff_bnd_B2 = diff_bnd_B**2

    # Calculate total loss
    loss = loss_F.mean() + loss_R.mean() 
    loss_wBnd = loss + diff_bnd_A2 + diff_bnd_B2 
    return loss_wBnd, loss

def write_traj (fname, traj_xyz_nm):
    ### Import DCD file and write trajectory
    #   write the data in Angstorm format
    ### write into a new DCD file in 'fname' variable
    from mdtraj.formats import DCDTrajectoryFile

    # Convert the input trajectory to Angstroms
    traj_xyz = traj_xyz_nm*10 ### why don't you use nm2ang???

    # Get the number of configurations in the trajectory
    n_conf = traj_xyz.shape[0]

    # If the last dimension of traj_xyz is not 3, reshape the array
    if traj_xyz.shape[-1] != 3:
        traj_xyz = traj_xyz.reshape(n_conf, -1, 3)
    
    # Open a new DCD trajectory file for writing
    with DCDTrajectoryFile(fname, 'w') as f:
        f.write (traj_xyz)
        
        
def get_trajectory (fname_prmtop, fname_dcd, nsamp):
    ### load and return last 'nsamp' frame
    #   Consistency in unit calculation
    ### jax_amber.py also uses nm to calculate 
    import mdtraj as md

    c = md.load (fname_dcd, top=fname_prmtop)
    crds = jnp.array (c.xyz)
    return crds[-nsamp:], crds[:-nsamp] # in nm unit
