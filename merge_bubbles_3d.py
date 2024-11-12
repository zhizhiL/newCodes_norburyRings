import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib.patches as patches

# load geometry files
path = 'velocity_results/alpha04_2D_'

geometry = np.load(path + 'geometry.npy', allow_pickle=True)
x_core, y_core, y_core_lower, x_ring, y_ring = geometry.T


def merge_package(Bubbles_df_before_merge: np.ndarray, advected_states: np.ndarray, 
                  gridA_size: tuple, gridB_size: tuple, 
                  boundaries: tuple, cell_size: tuple, R_collision: float,
                  merge_method:str, timeNow: float, this_ax , color) -> np.ndarray:
    
    """
        Merge bubbles that are close to each other
        :param bubble_df_ini: bubble dataframe before merging
        :param gridA_size: number of cells of grid A, (Ny_gridA, Nx_gridA)
        :param gridB_size: number of cells of grid B, (Ny_gridB, Nx_gridB)
        :param boundaries: boundaries of the domain, (xl, xr, yd, yu)
        :param cell_size: size of the cells, (dx_col, dy_col)
        :param R_collision: critical collision radius

        for plotting purpose:
        :param a: radius of the vortex ring
        :param timeNow: current time

        :return: bubble dataframe after merging
    """

    if merge_method not in ['simple', 'volume-weighted']:
        raise ValueError('merge_method must be either simple or volume-weighted')

    ## unpack the arguments
    xl, xr, yl, yr, zd, zu = boundaries
    dx_col, dy_col, dz_col = cell_size

    def put_bubbles_in_cell(bubbles_df: np.ndarray):
        '''
        inputs: bubbles_df after advection
        modifies the cell index of the bubbles dataframe in place
        '''
        bubbles_df[:, 9] = np.floor((bubbles_df[:, 1] - xl ) / dx_col).astype(int)
        bubbles_df[:, 10] = np.floor((bubbles_df[:, 2] - yl ) / dy_col).astype(int)
        bubbles_df[:, 11] = np.floor((bubbles_df[:, 3] - zd ) / dz_col).astype(int)
        bubbles_df[:, 12] = np.floor((bubbles_df[:, 1]- xl + dx_col / 2) / dx_col).astype(int)
        bubbles_df[:, 13] = np.floor((bubbles_df[:, 2]- yl + dy_col / 2) / dy_col).astype(int)
        bubbles_df[:, 14] = np.floor((bubbles_df[:, 3]- zd + dz_col / 2) / dz_col).astype(int)

        return bubbles_df
    
    def merge_bubbles(Bubbles_df_ini: np.ndarray) -> np.ndarray:

        Bubbles_df_new = Bubbles_df_ini.copy()
        
        Fbub_A = np.zeros(gridA_size)
        Fbub_B = np.zeros(gridB_size)
        drawer_A = {}
        drawer_B = {}
        masters_slaves_dict = {}

        # initialize the bubble field and the drawers based on the initial bubble distribution

        for i in range(len(Bubbles_df_ini)):
            Fbub_A[Bubbles_df_ini[i, 9].astype(int), Bubbles_df_ini[i, 10].astype(int), Bubbles_df_ini[i, 11].astype(int)] += 1
            Fbub_B[Bubbles_df_ini[i, 12].astype(int), Bubbles_df_ini[i, 13].astype(int), Bubbles_df_ini[i, 14].astype(int)] += 1

            if (Bubbles_df_ini[i, 9].astype(int), Bubbles_df_ini[i, 10].astype(int), Bubbles_df_ini[i, 11].astype(int)) not in drawer_A:
                drawer_A[(Bubbles_df_ini[i, 9].astype(int), Bubbles_df_ini[i, 10].astype(int), Bubbles_df_ini[i, 11].astype(int))] = [Bubbles_df_ini[i, 0]]
            else:
                drawer_A[(Bubbles_df_ini[i, 9].astype(int), Bubbles_df_ini[i, 10].astype(int), Bubbles_df_ini[i, 11].astype(int))].append(Bubbles_df_ini[i, 0])

            if (Bubbles_df_ini[i, 12].astype(int), Bubbles_df_ini[i, 13].astype(int), Bubbles_df_ini[i, 14].astype(int)) not in drawer_B:
                drawer_B[(Bubbles_df_ini[i, 12].astype(int), Bubbles_df_ini[i, 13].astype(int), Bubbles_df_ini[i, 14].astype(int))] = [Bubbles_df_ini[i, 0]]
            else:
                drawer_B[(Bubbles_df_ini[i, 12].astype(int), Bubbles_df_ini[i, 13].astype(int), Bubbles_df_ini[i, 14].astype(int))].append(Bubbles_df_ini[i, 0])

        
        def find_partner_in_cell(xj:int, yi:int, zk:int, bubble_ID, drawer: dict, bubbles_df: np.ndarray):
            '''
            Find the closest particle in the cell (xj, yi, zk) to the particle bubble_ID, 
            using the updated dataframe
            '''

            # collect all bubbles in the same cell, given by neighbors ID
            neighbors = drawer[(xj, yi, zk)]

            if len(neighbors) == 1:
                return 20*(xr-xl), None
            
            else:
                # calculate the distance between the bubble and all the neighbors in cell
                neighbor_rows = bubbles_df[np.isin(bubbles_df[:, 0], neighbors)]
                bubble_row = bubbles_df[np.where(bubbles_df[:, 0] == bubble_ID)[0][0]]

                kd_tree = sp.spatial.KDTree(neighbor_rows[:, 1:4])

                min_dist, min_index = kd_tree.query(bubble_row[1:4], k=[2])

                partner_ID = neighbor_rows[min_index[0], 0]

                return min_dist, partner_ID
            

        def update_newly_merged(master_ID, slave_ID, bubbles_df: np.ndarray, masters_slaves_dict: dict):
            '''
            modify the input bubble dataframe in place and the masters_slaves_dict
            '''
            master_row = bubbles_df[np.where(bubbles_df[:, 0] == master_ID)[0][0], :]
            slave_row = bubbles_df[np.where(bubbles_df[:, 0] == slave_ID)[0][0], :]

            # eliminate both bubbles from fields
            Fbub_A[master_row[9].astype(int), master_row[10].astype(int), master_row[11].astype(int)] -= 1
            Fbub_B[master_row[12].astype(int), master_row[13].astype(int), master_row[14].astype(int)] -= 1
            Fbub_A[slave_row[9].astype(int), slave_row[10].astype(int), slave_row[11].astype(int)] -= 1
            Fbub_B[slave_row[12].astype(int), slave_row[13].astype(int), slave_row[14].astype(int)] -= 1

            # eliminate both bubbles from drawers
            drawer_A[(master_row[9].astype(int), master_row[10].astype(int), master_row[11].astype(int))].remove(master_ID)
            drawer_B[(master_row[12].astype(int), master_row[13].astype(int), master_row[14].astype(int))].remove(master_ID)
            drawer_A[(slave_row[9].astype(int), slave_row[10].astype(int), slave_row[11].astype(int))].remove(slave_ID)
            drawer_B[(slave_row[12].astype(int), slave_row[13].astype(int), slave_row[14].astype(int))].remove(slave_ID)

            # update master_slave_dict
            masters_slaves_dict[master_ID] = slave_ID

            # update the new stokes number
            new_st = (master_row[7]**1.5 + slave_row[7]**1.5)**(2/3)

            # update the bubble dataframe location and velocity using volume-weighted average
            new_xp, new_yp, new_zp, new_vx, new_vy, new_vz = (master_row[1:7] * (master_row[7]**1.5) + 
                                                            slave_row[1:7] * (slave_row[7]**1.5)) / ((master_row[7]**1.5) + (slave_row[7]**1.5))
            
            new_jA = np.floor((new_xp - xl ) / dx_col).astype(int)
            new_jB = np.floor((new_xp - xl + dx_col / 2) / dx_col).astype(int)
            new_iA = np.floor((new_yp - yl ) / dy_col).astype(int)
            new_iB = np.floor((new_yp - yl + dy_col / 2) / dy_col).astype(int)
            new_kA = np.floor((new_zp - zd ) / dz_col).astype(int)
            new_kB = np.floor((new_zp - zd + dz_col / 2) / dz_col).astype(int)

            # update the master_row and slave_row
            bubbles_df[np.where(bubbles_df[:, 0] == master_ID)[0][0], :] = np.array([master_ID, new_xp, new_yp, new_zp, 
                                                                                    new_vx, new_vy, new_vz, new_st, False, new_jA, new_iA, new_kA, new_jB, new_iB, new_kB])
            
            bubbles_df[np.where(bubbles_df[:, 0] == slave_ID)[0][0], :] = np.full(slave_row.shape, np.nan)

            # update the fields
            Fbub_A[new_jA, new_iA, new_kA] += 1
            Fbub_B[new_jB, new_iB, new_kB] += 1

            drawer_A[(new_jA, new_iA, new_kA)].append(master_ID)
            drawer_B[(new_jB, new_iB, new_kB)].append(master_ID)

            return (new_xp, new_yp, new_zp)
        

        collision_point_list = []


        for i in range(len(Bubbles_df_new)):

            # if slaved, skip
            if Bubbles_df_new[i, 8]:
                pass

            else:
                bubble = Bubbles_df_new[i]
                jA, iA, kA = bubble[9:12].astype(int)
                jB, iB, kB = bubble[12:15].astype(int)

                # find the closest bubble in the two cells
                rmin_A, partner_ID_A = find_partner_in_cell(jA, iA, kA, bubble[0], drawer_A, Bubbles_df_new)
                rmin_B, partner_ID_B = find_partner_in_cell(jB, iB, kB, bubble[0], drawer_B, Bubbles_df_new)

                # if the minimal distance is larger than the collision radius, skip
                if min(rmin_A, rmin_B) > R_collision:
                    pass

                # if no partner in either cell, skip
                elif partner_ID_A is None and partner_ID_B is None:
                    pass

                else:
                    if rmin_A < rmin_B:
                        partner_ID = partner_ID_A

                    else:
                        partner_ID = partner_ID_B
                    
                    # now perform collision, the one with smaller ID is slaved
                    if bubble[0] < partner_ID:
                        collision_point = update_newly_merged(master_ID=partner_ID, slave_ID=bubble[0], 
                                                            bubbles_df=Bubbles_df_new, masters_slaves_dict=masters_slaves_dict)
                    
                    else:
                        collision_point = update_newly_merged(master_ID=bubble[0], slave_ID=partner_ID, 
                                                            bubbles_df=Bubbles_df_new, masters_slaves_dict=masters_slaves_dict)
                    
                    collision_point_list.append(collision_point)

        update_bubbles_df = Bubbles_df_new[~np.isnan(Bubbles_df_new[:, 1])]

        # Plotting
        fig, ax = plt.subplots()
        ax = fig.add_subplot(111, projection='3d')
        marker_size =  (update_bubbles_df[:, 7] * 199/1.4 - 185/14)/10
        # inside = update_bubbles_df[:, 1] **2 + update_bubbles_df[:, 2] **2 + update_bubbles_df[:, 3] **2 < 4
        inside = np.ones_like(update_bubbles_df[:, 1], dtype=bool)

        ax.scatter3D(update_bubbles_df[inside, 1], update_bubbles_df[inside, 2], update_bubbles_df[inside, 3], 
                   s=marker_size[inside]**0.5, c='k', marker='o', alpha=0.5, linewidths=0)

        ax.scatter3D(x_core, np.zeros_like(x_core), y_core, c='r', s=10, alpha=0.3)
        ax.scatter3D(x_core, np.zeros_like(x_core), y_core_lower, c='r', s=10, alpha=0.3)
        ax.scatter3D(x_ring, np.zeros_like(x_ring), y_ring, c='b', s=10, alpha=0.3)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        # Set perspective projection
        ax.set_proj_type('persp')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([3,3,3])

        ax.set_title('Bubbles at time = {:.3f}'.format(timeNow))

        return update_bubbles_df, collision_point_list
    
    # use advected properties to update the bubble dataframe
    Bubbles_df_before_merge[:, 1:7] = advected_states[:, 0:6]

    Bubbles_df_before_merge = put_bubbles_in_cell(Bubbles_df_before_merge)


    # check if any bubbles are out of the domain, if so, remove them
    Bubbles_df_before_merge = Bubbles_df_before_merge[(Bubbles_df_before_merge[:, 1] >= xl) &
                                                      (Bubbles_df_before_merge[:, 1] < xr) &
                                                      (Bubbles_df_before_merge[:, 2] >= yl) &
                                                      (Bubbles_df_before_merge[:, 2] < yr) &
                                                      (Bubbles_df_before_merge[:, 3] >= zd) &
                                                      (Bubbles_df_before_merge[:, 3] < zu)]
    
    # merge bubbles
    Bubbles_df_after_merge, collision_point_list = merge_bubbles(Bubbles_df_before_merge)

    return Bubbles_df_after_merge, collision_point_list