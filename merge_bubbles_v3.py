import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib.patches as patches





def merge_package(Bubbles_df_before_merge: np.ndarray, advected_states: np.ndarray, 
                  gridA_size: tuple, gridB_size: tuple, 
                  boundaries: tuple, cell_size: tuple, R_collision: float, st_lim: float,
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
    xl, xr, yd, yu = boundaries
    dx_col, dy_col = cell_size
    # collision_box = np.zeros(gridA_size)



    def put_bubbles_in_cell(bubbles_df):
            '''
            modify the cell index columns of the bubbles dataframe in place
            '''
            bubbles_df[:, 7] = np.floor((bubbles_df[:, 1] - xl ) / dx_col).astype(int)
            bubbles_df[:, 8] = np.floor((bubbles_df[:, 2] - yd ) / dy_col).astype(int)
            bubbles_df[:, 9] = np.floor((bubbles_df[:, 1]- xl + dx_col / 2) / dx_col).astype(int)
            bubbles_df[:, 10] = np.floor((bubbles_df[:, 2]- yd + dy_col / 2) / dy_col).astype(int)

            return bubbles_df


    def merge_bubbles(Bubbles_df_ini: np.ndarray)-> np.ndarray:


        Bubbles_df_new = Bubbles_df_ini.copy()
        xl, xr, yd, yu = boundaries
        dx_col, dy_col = cell_size

        Fbub_A = np.zeros(gridA_size)
        Fbub_B = np.zeros(gridB_size)
        drawer_A = {}
        drawer_B = {}
        masters_slaves_dict = {}

        # initialize the bubble field and the drawers based on the initial bubble distribution

        for i in range(len(Bubbles_df_ini)):
            Fbub_A[Bubbles_df_ini[i, 8].astype(int), Bubbles_df_ini[i, 7].astype(int)] += 1
            Fbub_B[Bubbles_df_ini[i, 10].astype(int), Bubbles_df_ini[i, 9].astype(int)] += 1

            if (Bubbles_df_ini[i, 8].astype(int), Bubbles_df_ini[i, 7].astype(int)) not in drawer_A:
                drawer_A[(Bubbles_df_ini[i, 8].astype(int), Bubbles_df_ini[i, 7].astype(int))] = [Bubbles_df_ini[i, 0]]
            else:
                drawer_A[(Bubbles_df_ini[i, 8].astype(int), Bubbles_df_ini[i, 7].astype(int))].append(Bubbles_df_ini[i, 0])

            if (Bubbles_df_ini[i, 10].astype(int), Bubbles_df_ini[i, 9].astype(int)) not in drawer_B:
                drawer_B[(Bubbles_df_ini[i, 10].astype(int), Bubbles_df_ini[i, 9].astype(int))] = [Bubbles_df_ini[i, 0]]
            else:
                drawer_B[(Bubbles_df_ini[i, 10].astype(int), Bubbles_df_ini[i, 9].astype(int))].append(Bubbles_df_ini[i, 0])
        

        def find_partner_in_cell(xj, yi, bubble_ID:int, drawer:dict, bubbles_df):

            '''
            Find the closest particle in the cell (xj, yi) to the particle bubble_ID, 
            using the updated bubble dataframe
            '''

            # collect all particles in the cell, given by neighbors ID
            neighbors = drawer[(yi, xj)]

            # dont consider the neighbors that has St > 7, so that the largest merged St is 10:
            neighbors = [neigh for neigh in neighbors if bubbles_df[np.where(bubbles_df[:, 0] == neigh)[0][0], 5] < st_lim]

            if len(neighbors) == 1:
                
                return 10*(xr-xl), None
            
            else:
                # calculate the distance between the particle and all other particles in the cell
                neighbor_rows = bubbles_df[np.isin(bubbles_df[:, 0], neighbors)]
                bubble_row = bubbles_df[np.where(bubbles_df[:, 0] == bubble_ID)[0][0]]

                kd_tree = sp.spatial.KDTree(neighbor_rows[:, 1:3])

                min_dist, min_idx = kd_tree.query(bubble_row[1:3], k=[2]) 

                partner_ID = neighbor_rows[min_idx[0], 0]

                return min_dist, partner_ID


        def update_newly_merged(master_ID:float, slave_ID:float, bubbles_df:np.ndarray, masters_slaves_dict):

            '''
            modify the input bubble dataframe in place and the masters_slaves_dict
            '''

            master_row = bubbles_df[np.where(bubbles_df[:, 0] == master_ID)[0][0], :]
            slave_row = bubbles_df[np.where(bubbles_df[:, 0] == slave_ID)[0][0], :]

            # eliminate both bubbles from the fields
            Fbub_A[slave_row[8].astype(int), slave_row[7].astype(int)] -= 1
            Fbub_B[slave_row[10].astype(int), slave_row[9].astype(int)] -= 1
            Fbub_A[master_row[8].astype(int), master_row[7].astype(int)] -= 1
            Fbub_B[master_row[10].astype(int), master_row[9].astype(int)] -= 1

            # eliminate both bubbles from the drawers
            drawer_A[(slave_row[8].astype(int), slave_row[7].astype(int))].remove(slave_ID)
            drawer_B[(slave_row[10].astype(int), slave_row[9].astype(int))].remove(slave_ID)
            drawer_A[(master_row[8].astype(int), master_row[7].astype(int))].remove(master_ID)
            drawer_B[(master_row[10].astype(int), master_row[9].astype(int))].remove(master_ID)

            # update master_slave_dict
            masters_slaves_dict[master_ID] = slave_ID

            
            # need to consider the assumption for merged stokes number more carefully
            new_st = (master_row[5]**1.5 + slave_row[5]**1.5)**(2/3)
            

            # update bubbles dataframe (simple averaging)
            if merge_method == 'simple':
                new_xp = (master_row[1] + slave_row[1]) / 2
                new_yp = (master_row[2] + slave_row[2]) / 2
                new_vx = (master_row[3] + slave_row[3]) / 2
                new_vy = (master_row[4] + slave_row[4]) / 2

            # update bubbles dataframe (volume-weighted averaging)
            elif merge_method == 'volume-weighted':
                new_xp = (master_row[1] * (master_row[5]**1.5) + slave_row[1] * (slave_row[5]**1.5)) / (new_st**1.5)
                new_yp = (master_row[2] * (master_row[5]**1.5) + slave_row[2] * (slave_row[5]**1.5)) / (new_st**1.5)
                new_vx = (master_row[3] * (master_row[5]**1.5) + slave_row[3] * (slave_row[5]**1.5)) / (new_st**1.5)
                new_vy = (master_row[4] * (master_row[5]**1.5) + slave_row[4] * (slave_row[5]**1.5)) / (new_st**1.5)


            # need to consider the assumption for merged stokes number more carefully
            new_st = (master_row[5]**1.5 + slave_row[5]**1.5)**(2/3)

            new_jA = np.floor((new_xp - xl ) / dx_col).astype(int)
            new_iA = np.floor((new_yp - yd ) / dy_col).astype(int)
            new_jB = np.floor((new_xp - xl + dx_col / 2) / dx_col).astype(int)
            new_iB = np.floor((new_yp - yd + dy_col / 2) / dy_col).astype(int)

            bubbles_df[np.where(bubbles_df[:, 0] == master_ID)[0][0], :] = np.array([master_ID, new_xp, new_yp, new_vx, new_vy, new_st, False, new_jA, new_iA, new_jB, new_iB])
            bubbles_df[np.where(bubbles_df[:, 0] == slave_ID)[0][0], :] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, True, np.nan, np.nan, np.nan, np.nan])

            # update bubble field
            Fbub_A[new_iA, new_jA] += 1
            Fbub_B[new_iB, new_jB] += 1

            drawer_A[(new_iA, new_jA)].append(master_ID)
            drawer_B[(new_iB, new_jB)].append(master_ID)

            # collision_box[new_iA, new_jA] += 1

            # return bubbles_df
            return (new_xp, new_yp), masters_slaves_dict
        
        collision_point_list = []
        # masters_slaves_list = []

        for i in range(len(Bubbles_df_new)):

            # if i in masters_slaves_dict or Bubbles_df_ini[i, 6] == True:
            if  Bubbles_df_new[i, 6] == True:
                pass

            ### bubble size limiter ###
            elif Bubbles_df_new[i, 5] > st_lim:
                pass

            else:
                bubble = Bubbles_df_new[i]
                iA, jA = bubble[8].astype(int), bubble[7].astype(int)
                iB, jB = bubble[10].astype(int), bubble[9].astype(int)

                ### remove the St > 10 as potential partners in respective drawers ###
                rmin_A, partner_ID_A = find_partner_in_cell(jA, iA, bubble[0], drawer_A, Bubbles_df_new)
                rmin_B, partner_ID_B = find_partner_in_cell(jB, iB, bubble[0], drawer_B, Bubbles_df_new)

                # if the minimal distance is larger than the collision radius, then no collision happens
                if min(rmin_A, rmin_B) > R_collision:
                    pass

                # if no neighbour in either cell, then no collision happens
                elif partner_ID_A is None and partner_ID_B is None:
                    pass

                else:
                    if rmin_A < rmin_B:
                        partner_ID = partner_ID_A

                    else:
                        partner_ID = partner_ID_B
                    
                    # now perform the collision
                    slaved = bubble[0] < partner_ID

                    if slaved:
                        collision_point, this_dict = update_newly_merged(master_ID=partner_ID, slave_ID=bubble[0], bubbles_df=Bubbles_df_new, 
                                                                         masters_slaves_dict=masters_slaves_dict)

                    else:
                        collision_point, this_dict = update_newly_merged(master_ID=bubble[0], slave_ID=partner_ID, bubbles_df=Bubbles_df_new, 
                                                              masters_slaves_dict=masters_slaves_dict)
                    
                    collision_point_list.append(collision_point)
                    # masters_slaves_list.append(this_dict)

        update_bubbles_df = Bubbles_df_new[~np.isnan(Bubbles_df_new[:, 1])]

        # plot the updated bubbles
        path = 'velocity_results/alpha04_2D_'
        geometry = np.load(path + 'geometry.npy')
        x_core, y_core, y_core_lower, x_ring, y_ring = geometry.T

        fig, ax = plt.subplots()
        this_ax = ax
        marker_size = (update_bubbles_df[:, 5] * 199/1.4 - 185/14)/10
        this_ax.scatter(update_bubbles_df[:, 1], update_bubbles_df[:, 2], s=marker_size**0.5, marker='o', c=color, linewidths=0)
        this_ax.plot(x_core, y_core, 'k', x_core, y_core_lower, 'k',lw=1, alpha=0.6)
        this_ax.plot(x_ring, y_ring, 'k', lw=1, alpha=0.6)
        this_ax.set_xlim([xl, xr])
        this_ax.set_ylim([yd, yu])
        this_ax.set_aspect('equal')
        this_ax.set_title('{}-average Coalescence at t = {:.3f}'.format(merge_method, timeNow))
        
        # return update_bubbles_df
        return update_bubbles_df, collision_point_list, masters_slaves_dict

    # Bubbles_df_before_merge = Bubbles_df_to_adv.copy()
    Bubbles_df_before_merge[:, 1] = advected_states[:, 0]
    Bubbles_df_before_merge[:, 2] = advected_states[:, 1]
    Bubbles_df_before_merge[:, 3] = advected_states[:, 2]
    Bubbles_df_before_merge[:, 4] = advected_states[:, 3]

    Bubbles_df_before_merge = put_bubbles_in_cell(Bubbles_df_before_merge)

    # need to check if any bubbles are out of the domain, if so, remove them
    Bubbles_df_before_merge = Bubbles_df_before_merge[(Bubbles_df_before_merge[:, 1] >= xl) & (Bubbles_df_before_merge[:, 1] < xr) 
                                    & (Bubbles_df_before_merge[:, 2] >= yd) & (Bubbles_df_before_merge[:, 2] < yu)]

    # # merge_bubbles
    Bubbles_df_after_merge, collision_point_list, masters_slaves_list = merge_bubbles(Bubbles_df_before_merge)

    return Bubbles_df_after_merge, collision_point_list, masters_slaves_list 