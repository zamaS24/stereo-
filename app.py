import dearpygui.dearpygui as dpg

import numpy as np
from algorithms import Algorithm as algo

import matplotlib.pyplot as plt
import open3d as o3d


def _hsv_to_rgb(h, s, v):
    if s == 0.0: return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
    if i == 0: return (255*v, 255*t, 255*p)
    if i == 1: return (255*q, 255*v, 255*p)
    if i == 2: return (255*p, 255*v, 255*t)
    if i == 3: return (255*p, 255*q, 255*v)
    if i == 4: return (255*t, 255*p, 255*v)
    if i == 5: return (255*v, 255*p, 255*q)

class App: 
    def __init__(self):
        
        # set the font
        with dpg.font_registry():
            default_font = dpg.add_font(
                "resources/fonts/Roboto_Mono/static/RobotoMono-Regular.ttf",28
            )
        dpg.bind_font(default_font)
    
        
        # Back-end attributes 
        self.imageLPath = None
        self.imageRPath = None
        self.directoryPath = None
        self.distortion = None

        self.matrix_intrinsinc = None
        self.image_matches = None
        self.camera_coords = None


        # State attributes 
        self.fileIsLoaded = False

        
        
        with dpg.theme(tag="__demo_theme"+str(1)):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(5/7.0, 0.6, 0.6))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(5/7.0, 0.8, 0.8))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(5/7.0, 0.7, 0.7))
                
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 1*5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 1*3, 1*3)

        # Main window here
        with dpg.window(
            label="Simple Stereo System",
            no_resize=True, no_close=True,
            no_open_over_existing_popup=True,
            width=1080, height=1050, 
            no_collapse=True, no_move=True 
        ) as self.wdw_main:

            with dpg.collapsing_header(
                tag ="calibration_header", 
                label="Step1: Calibrer la caméra"
            ) as self.chc:
                
                dpg.add_text(
                    "Veuillez, prendre plusieurs images d'un échiquier de 7*9"
                    " sous format png pour calibrer votre camera, ensuite" 
                    " introduire le chemin vers le dossier contenant ces images.",
                    wrap=0
                )
                
                with dpg.file_dialog(
                    directory_selector=True, 
                    show=False, 
                    callback=self.callback_folder_dialog,
                    file_count=3,  
                    width=700 ,height=450
                ) as self.folderDialog:
                    pass


                with dpg.group(horizontal=True):

                    self.btn_file = dpg.add_button(
                        label="Selectionner le dossier",
                        callback=lambda: dpg.show_item(self.folderDialog) 
                    )
                    dpg.bind_item_theme(dpg.last_item(), "__demo_theme"+str(1))

                    self.btn_execute_calibration = dpg.add_button(
                        label="Calibrer", 
                        callback = self.callback_calibrer
                    )
                    dpg.bind_item_theme(dpg.last_item(), "__demo_theme"+str(1))
                    dpg.add_loading_indicator(show=False, tag="loading1")


            # Resultat de la calibration
            with dpg.collapsing_header(
                tag ="mtx",
                label="Matrice de calibration", 
                show=False
            ):
                dpg.add_text("Matrice intrinsinc")
           
            dpg.add_spacer(height=10)

            with dpg.collapsing_header(
                tag="step2",
                label=" Step2 - Introduire L'image Left et l'image right", 
                show=False
            ) as self.ch3d: 

                # Ajouter un `file dialog` pour image left
                with dpg.file_dialog(
                    directory_selector=False, 
                    show=False, 
                    callback=self.callback_file_dialog_L, 
                    file_count=3,  
                    width=700 ,height=450
                ) as self.fileDialogL:
                    dpg.add_file_extension(".jpg", color=(255, 255, 0, 255))

                # Ajouter un `file dialog` pour image Right
                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=self.callback_file_dialog_R, 
                    file_count=3,  
                    width=700 ,height=450
                ) as self.fileDialogR:
                    dpg.add_file_extension(".jpg", color=(255, 255, 0, 255))

                # Button to image Left path 
                self.btn_file_L = dpg.add_button(
                    label="Selectionnez chemin vers image gauche",  
                    callback= lambda : dpg.show_item(self.fileDialogL)
                )
                dpg.bind_item_theme(dpg.last_item(), "__demo_theme"+str(1))

                # Button to image Right path
                self.btn_file_R = dpg.add_button(
                    label="Selectionnez chemin vers  image droite ",  
                    callback=lambda : dpg.show_item(self.fileDialogR)
                )
                dpg.bind_item_theme(dpg.last_item(), "__demo_theme"+str(1))

                # Input float pour la disparité  : 
                self.btn_input_distortion = dpg.add_input_float(
                    label="Donner la disparité en cm", 
                    callback=self.callback_input_distortion, 
                    format="%.06f"
                )

                dpg.add_spacer(height=30)

                # Execute Depth estimation process              
                with dpg.group(horizontal=True):
                    self.btn_execute = dpg.add_button(
                        tag ="execute", 
                        label="Estimer les points 3D", 
                        show=False, 
                        callback=self.callback_estimation3D
                    )
                    dpg.bind_item_theme(dpg.last_item(), "__demo_theme"+str(1))
                    dpg.add_loading_indicator(show=False, tag="loading2")   
                                 
            dpg.add_spacer(height=10)        

            # resultats POINTS SIFT et Depth estimation       
            with dpg.collapsing_header(
                tag ="step3", 
                label="Points SIFT et coordonnés 3D", 
                show=False
            ):

                dpg.add_button(
                    label="Afficher les points SIFTs appareillés", 
                    callback=self.callback_afficher_SIFTmatch
                )           
                dpg.bind_item_theme(dpg.last_item(), "__demo_theme"+str(1))

                dpg.add_button(
                    label="Afficher les coordonnées 3d", 
                    callback=self.callback_show_3d_coords
                )
                dpg.bind_item_theme(dpg.last_item(), "__demo_theme"+str(1))

                # Pour afficher le nuage des points 
                dpg.add_button(
                    label="Sauvegarder le nuage des points", 
                    callback=self.callback_show_points_cloud
                )
                dpg.bind_item_theme(dpg.last_item(), "__demo_theme"+str(1))

            # self.btn_test = dpg.add_button(label="test", callback = self.callback_test)

        with dpg.window(
            label="Coordonnées 3D", 
            pos=(1090,0), 
            width=800, height=1050, 
            show=False
        ) as self.wdw_3d:
            with dpg.collapsing_header(
                tag ="3d_coords",
                label="Coordonnes 3d relative a la camera", 
                show=False
            ):
                pass



    # -------------- Callbacks 

    # NOTE: use this function for debugging purposes
    def callback_test(self,sender, app_data, user_data):
        pass

    def callback_input_distortion(self, sender, app_data):
        self.distortion = app_data
        print(f"La distortion entrée par l'utilisateur = {self.distortion}")
         
    def callback_file_dialog_L(self, sender, app_data): 
        self.imageLPath = app_data['file_path_name']
        if self.imageLPath: 
            print("File path to image Left : ", self.imageLPath)

    def callback_file_dialog_R(self, sender, app_data): 
        self.imageRPath = app_data['file_path_name']
        if self.imageRPath: 
            print("File path to image Right: ", self.imageRPath)
       
    def callback_folder_dialog(self,sender, app_data):
            self.directoryPath = app_data['file_path_name']
            if self.directoryPath: 
                print("Directory path to calibration folder: ", self.directoryPath)
                # TODO: Afficher les images contenu dans ce folder dans une fenetre a part 

    def callback_afficher_SIFTmatch(self, sender, app_data): 
        dpg.delete_item(item="warning_sift_img")

        if self.image_matches is None:
            dpg.add_text(
                "Image non retournée! ", 
                tag="warning_sift_img", 
                parent=self.chc, 
                color=[255,0,0]
            )           
        
        if(self.image_matches is not None): 
            plt.imshow(self.image_matches)
            plt.title("Appareimment des points SIFTs")
            plt.show()  

    def callback_calibrer(self): 
        #Delete whatever or hide whatever item that is bein infected by this callback
        dpg.configure_item(item="mtx", show=False)
        dpg.delete_item(item ="Warning_folder_path")
        dpg.delete_item(item ="success_calibration")
        dpg.delete_item(item="calib_matrix")

        if not self.directoryPath:
            dpg.add_text(
                "Donner d'abord le chemin ! ", 
                tag="Warning_folder_path", 
                parent=self.chc, 
                color=[255,0,0]
            )
        else: 
            dpg.configure_item("loading1",show=True )
            self.matrix_intrinsinc = algo.calibrer_camera(self.directoryPath)

            dpg.add_text(
                "Résultat ici  ", 
                tag="success_calibration", 
                parent=self.chc, 
                color=[0,255,0]
            )
            dpg.configure_item("step2",show=True )
            dpg.configure_item(self.btn_execute,show=True )

            with dpg.table(
                parent="mtx", 
                tag="calib_matrix", 
                header_row=False, row_background=True,
                borders_innerH=True, borders_outerH=True,
                borders_innerV=True, borders_outerV=True,
                delay_search=True
            ) as table_id:

                dpg.add_table_column(label="Header 1")
                dpg.add_table_column(label="Header 2")
                dpg.add_table_column(label="Header 3")

                for i in range(3):
                    with dpg.table_row():
                        for j in range(3):
                            dpg.add_text(f"{self.matrix_intrinsinc[i][j]}")
                
                for i in range(3):
                        dpg.highlight_table_row(table_id, i, [0, 255, 0, 100])
            
            dpg.configure_item(item="mtx", show=True)
            dpg.configure_item("loading1",show=False )
    
    def callback_estimation3D(self):

        # Managing GUI STUFF 
        dpg.delete_item(item ="Warning_imageL_path")
        dpg.delete_item(item ="Warning_imageR_path")
        dpg.delete_item(item ="Warning_distortion")
        dpg.delete_item(item ="success_sift_3d")
        # Managing errors
        if not self.imageLPath:
            dpg.add_text(
                "Donner d'abord le chemin de l'image Left! ", 
                tag="Warning_imageL_path", 
                parent=self.ch3d, 
                color=[255,0,0]
            )
        
        if not self.imageRPath:
            dpg.add_text(
                "Donner d'abord le chemin de l'image Right! ", 
                tag="Warning_imageR_path", 
                parent=self.ch3d, 
                color=[255,0,0]
            )
        
        if self.distortion is None or self.distortion <=0:
            dpg.add_text(
                "Donner une valeure positive de la distortion  ", 
                tag="Warning_distortion", 
                parent=self.ch3d, 
                color=[255,0,0]
            )
         
        # Actual Algorithm
        if self.imageLPath and self.imageRPath and self.distortion > 0:

            dpg.configure_item("loading2",show=True)
            self.image_matches, self.camera_coords = algo.calculer_pts3D(
                self.matrix_intrinsinc, 
                self.distortion, 
                self.imageLPath, 
                self.imageRPath
            )
            dpg.configure_item("loading2", show=False)

            #Unlock next steps 
            dpg.add_text(
                "Résultat ici  ", 
                tag="success_sift_3d", 
                parent=self.ch3d, 
                color=[0,255,0]
            )
            dpg.configure_item(item="step3", show=True)
            #Afficher les coordonnées 3d 


            # Afficher l'image des points SIFT 
            #Create a texture from the image
            
    def callback_show_points_cloud(self): 
        points_arr = np.array(self.camera_coords)
        points_arr[:, 2] /= 10
        # Create an Open3D point cloud from the numpy array
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_arr)

        # Save the point cloud in PLY format
        o3d.io.write_point_cloud("point_cloud.ply", pcd)
       
    def callback_show_3d_coords(self, sender, app_data):
        dpg.configure_item(self.wdw_3d, show=True)
        dpg.delete_item(item="3d_coords", children_only=True)
        dpg.delete_item(item ="Warning_3d_coords")

        if self.camera_coords is None:
            dpg.add_text("Les coordonnées ne sont pas prets!", tag="Warning_3d_coords", parent=self.ch3d, color=[255,0,0])

        if(self.camera_coords is not None):
             dpg.configure_item(self.wdw_main, show=True)

             with dpg.table(parent="3d_coords", tag="table_3d_coords", header_row=False, row_background=True,
                        borders_innerH=True, borders_outerH=True, borders_innerV=True,
                        borders_outerV=True, delay_search=True) as table_id:

                    dpg.add_table_column(label="Header 1")
                    dpg.add_table_column(label="Header 2")
                    dpg.add_table_column(label="Header 3")

                    with dpg.table_row():
                        dpg.add_text("xc")
                        dpg.add_text("yc")
                        dpg.add_text("zc")

                    n = len(self.camera_coords)
                    for i in range(n):
                        with dpg.table_row():
                            for j in range(3):
                                dpg.add_text(f"{self.camera_coords[i][j]}")

                    dpg.configure_item("3d_coords", show=True)
             



# ----- Launch the app
dpg.create_context()

App()

dpg.create_viewport(
    width=1080, height=720, 
    resizable=False
)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()