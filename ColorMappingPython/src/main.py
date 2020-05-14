import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import rgb_munsell_mapping
from math import sin, cos
import math
# from matplotlib import style
# style.use('ggplot')
# import matplotlib
# matplotlib.use("TkAgg")
# import random
import numpy as np
from operator import itemgetter

from skimage import data


def get_rotation_matrix(axis, theta):
    u = []
    if axis == 'x':
        u = [1, 0, 0]
    elif axis == 'y':
        u = [0, 1, 0]
    elif axis == 'z':
        u = [0, 0, 1]

    return [[cos(theta) + u[0]**2 * (1-cos(theta)),
             u[0] * u[1] * (1-cos(theta)) - u[2] * sin(theta),
             u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta)],
            [u[0] * u[1] * (1-cos(theta)) + u[2] * sin(theta),
             cos(theta) + u[1]**2 * (1-cos(theta)),
             u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta)],
            [u[0] * u[2] * (1-cos(theta)) - u[1] * sin(theta),
             u[1] * u[2] * (1-cos(theta)) + u[0] * sin(theta),
             cos(theta) + u[2]**2 * (1-cos(theta))]]
def three_d_rotation(pointToRotate, rotation_matrix):
    r = rotation_matrix
    rotated = []

    for i in range(3):
        rotated.append(sum([r[j][i] * pointToRotate[j] for j in range(3)]))

    return rotated
def get_rotation_matrix_x_y(P1, P2):
    v1x = [P1[1], P1[2]]
    v2x = [P2[1], P2[2]]
    v1y = [P1[0], P1[2]]
    v2y = [P2[0], P2[2]]
    dir_x = -1 if v1x[0] * v2x[1] - v1x[1] * v2x[0] < 0 else 1
    dir_y = -1 if v1y[0] * v2y[1] - v1y[1] * v2y[0] < 0 else 1
    radx = dir_x * math.acos(round(np.dot(v1x, v2x) / (np.sqrt(np.dot(v1x, v1x)) * np.sqrt(np.dot(v2x, v2x))), 6))
    rady = dir_y * math.acos(round(np.dot(v1y, v2y) / (np.sqrt(np.dot(v1y, v1y)) * np.sqrt(np.dot(v2y, v2y))), 6))
    rx = get_rotation_matrix('y', rady)
    ry = get_rotation_matrix('x', radx)

    return rx, ry
def rotate_x(ox, px, oy, py, angle):
    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    return qx
def rotate_y(ox, px, oy, py, angle):
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    return qy

def new_window(root):
    new_win = tk.Toplevel(root)
    new_win.config(bg='white')
    new_win.geometry("{}x{}".format(SCREEN_WIDTH_RESULTS,SCREEN_HEIGHT_RESULTS))
    return new_win

class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
    def show(self):
        self.lift()
    def active(self, button):
        button.config(font=("Arial", 16, 'bold'))

class UploadImagePage(Page):
    def __init__(self, root, *args, **kwargs):
        self.path = {'path':''}
        self.root = root
        self.use_cropped_image = False


        Page.__init__(self, *args, **kwargs)

        self.first_row_frame = tk.Frame(self)
        self.first_row_frame.pack(side="top", pady=12)

        instruction_text = 'Please select an image to perform the Munsell color mapping'
        self.text_image_upload = tk.Label(self.first_row_frame, text=instruction_text,
                                         height=2, width=50, font=("Arial", 20))
        self.text_image_upload.pack()

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side="top")

        self.create_upload_button('Upload Image', 14)

        # self.im = rgb_munsell_mapping.generate_test_image()
        #
        # self.button_perform_mapping = tk.Button(self.first_row_frame, highlightbackground='red',
        #                                         text='Perform Mapping',
        #                                         font=("Courier", 15), command=lambda: [self.perform_mapping()])
        # self.button_perform_mapping.config(height=3, width=19, pady=1, relief=tk.RAISED)
        # self.button_perform_mapping.pack()

    def UploadAction(self):
        # Get local filename
        filename = filedialog.askopenfilename()
        self.path['path'] = filename

        # Create canvas to show image. Delete if there is an image
        try:
            self.canv.destroy()
        except:
            pass
        self.canv = tk.Canvas(self, bg='white')

        self.im = Image.open(filename)
        width, height = self.im.size
        self.canv_w = self.master.winfo_width()

        if width > IMG_WIDTH_MAX:
            scale = IMG_WIDTH_MAX/width
            self.im = self.im.resize((int(width*scale), int(height*scale)), Image.ANTIALIAS)
        width, height = self.im.size
        if height > IMG_HEIGHT_MAX:
            scale = IMG_HEIGHT_MAX/height
            self.im = self.im.resize((int(width*scale), int(height*scale)), Image.ANTIALIAS)


        self.update_buttons()
        self.canv.pack(side=tk.TOP, expand=True, fill='both')

        self.img = ImageTk.PhotoImage(self.im)
        self.imgtag = self.canv.create_image(self.canv_w/2, 0, anchor="n", image=self.img)

    def create_upload_button(self, text, width):
        try:
            self.button_image_upload.destroy()
        except:
            pass

        self.button_image_upload = tk.Button(self.button_frame, text=text,
                                             font=("Courier", 13), command=lambda: [self.UploadAction()])
        self.button_image_upload.config(height=3, width=width)
        self.button_image_upload.pack(side='left', padx=8)

    def create_crop_button(self):
        try:
            self.button_crop.destroy()
        except:
            pass
        self.button_crop = tk.Button(self.button_frame, text='Crop',
                                     font=("Courier", 13), command=lambda: [self.crop()])
        self.button_crop.config(height=3, width=8)
        self.button_crop.pack(side='left')

    def create_vectorization_level_menus(self):
        try:
            self.vectorization_level_scale.destroy()
        except:
            pass
        self.vectorization_level_scale = tk.Scale(self.first_row_frame, from_=1, to=10, orient='horizontal',
                                                  label='Threshold Level', length=120)
        self.vectorization_level_scale.pack(side='left', padx=8)


    def create_mapping_button(self):
        try:
            self.button_perform_mapping.destroy()
        except:
            pass

        try:
            self.text_image_upload.destroy()
        except:
            pass

        self.button_perform_mapping = tk.Button(self.first_row_frame, highlightbackground='red',
                                                text='Perform Mapping',
                                     font=("Courier", 15), command=lambda: [self.perform_mapping()])
        self.button_perform_mapping.config(height=3, width=19, pady=1, relief=tk.RAISED)
        self.button_perform_mapping.pack(side='right')

    def perform_mapping(self):
        self.results_window = new_window(root)
        self.root = root
        try:
            self.threshold_level = self.vectorization_level_scale.get()
        except AttributeError:
            self.threshold_level = 5

        # Setup Frame Organization
        self.results_model_frame = tk.Frame(self.results_window, borderwidth=0, relief="solid")
        self.results_model_frame.grid(row=0, columnspan=2)

        # Generate vectorized image
        self.image_canvas = tk.Canvas(self.results_model_frame, bg='white',
                                        width=IMG_WIDTH_MAX, height=IMG_HEIGHT_MAX,
                                        borderwidth=1, relief="solid")
        self.image_canvas.grid(row=0, column=0)

        if self.use_cropped_image == True:
            self.vectorized_raw, self.df, self.rgb_hvc_list = rgb_munsell_mapping.perform_mapping(self.cropped_im,
                                                                                                  self.threshold_level)
        else:
            self.vectorized_raw, self.df, self.rgb_hvc_list = rgb_munsell_mapping.perform_mapping(self.im,
                                                                                                  self.threshold_level)

        self.vectorized_im = ImageTk.PhotoImage(self.vectorized_raw)
        self.vectorized_img = self.image_canvas.create_image(IMG_WIDTH_MAX / 2, IMG_HEIGHT_MAX/2, anchor="c",
                                                             image=self.vectorized_im)

        # 3D Models
        self.three_d_model_canvas = tk.Canvas(self.results_model_frame, bg='white',
                                              width=IMG_WIDTH_MAX - 50, height=IMG_HEIGHT_MAX,
                                              borderwidth=1, relief="solid")
        self.three_d_model_canvas.grid(row=0, column=1)
        self.generate_3d_model()

        # 2D View
        self.two_d_model_canvas = tk.Canvas(self.results_model_frame, bg='white',
                                              width=IMG_WIDTH_MAX - 50, height=IMG_HEIGHT_MAX,
                                              borderwidth=1, relief="solid")
        self.two_d_model_canvas.grid(row=0, column=1)
        self.two_d_model_canvas.grid_forget()

        # Pixels information text
        text = 'Pixels information: '
        self.text_pixels_info = tk.Label(self.results_model_frame, text=text,
                                          height=2, width=15, font=("Arial", 16, 'bold'))
        self.text_pixels_info.grid(row=1, column=0, padx=5, sticky='nw')

        # Pixels information frame
        self.pixels_info_frame = tk.Frame(self.results_model_frame, borderwidth=0, relief="solid")
        self.pixels_info_frame.grid(row=2, column=0, sticky='nw')

        # 2D view button
        self.view_2d_page_button = tk.Button(self.results_model_frame, text='View Selected Page', state='disabled',
                                             font=("Courier", 15), command=self.view_2D)
        self.view_2d_page_button.config(height=3, width=19, padx=1, relief=tk.RAISED)
        self.view_2d_page_button.grid(row=1, column=1, padx=4, sticky='ne')

        # Activate bind
        self.highlight_mode = False
        self.image_canvas.tag_bind(self.vectorized_img, '<Button-1>', self.update_RGB_info_from_image_selection)
        self.image_canvas.tag_bind(self.vectorized_img, '<Motion>', self.change_cursor)
        self.image_canvas.tag_bind(self.vectorized_img, '<Leave>', self.revert_cursor)

    def generate_3d_model(self):
        self.cylinder_shapes = []
        self.cylinder_figures = []
        self.box_shapes = []
        self.box_figures = []

        self.hues = {'5R':0, '10R':1, '5YR':2, '10YR':3, '5Y':4, '10Y':5, '5GY':6, '10GY':7, '5G':8, '10G':9, '5BG': 10,
                     '10BG':11, '5B':12, '10B':13, '5PB':14, '10PB':15, '5P':16, '10P':17, '5RP':18, '10RP':19}

        # Get the color and 3D position of each object
        for idx, val in self.df.iterrows():
            r, g, b = val['r'], val['g'], val['b']
            hue = val['hue']
            value = val['value']
            chroma = val['chroma']
            if val['hue'][0] == 'N' and val['color_counts']>0:
                self.cylinder_shapes.append( ({'3dpts':[], '2dpts':[], 'figure_IDs':[]}, (r,g,b), value,chroma,hue) )
            elif val['hue'][0] != 'N' and val['color_counts'] > 0:
                angle = self.hues[val['hue']]
                page_ID = self.hues[val['hue']]
                if page_ID >= 10:
                    page_ID -= 10
                self.box_shapes.append( ({'midpts':[], '3dpts':[], '2dpts':[], 'page_ID':page_ID, 'figure_IDs':[]},
                                         (r,g,b), value, angle, chroma, hue) )

        # Setting
        self.root.update()
        self.canvas_3d_model_width = self.three_d_model_canvas.winfo_width()
        self.canvas_3d_model_height = self.three_d_model_canvas.winfo_height()
        self.cam_distance = 100
        self.ox = int(self.canvas_3d_model_width/2)
        self.oy = int(self.canvas_3d_model_height/2 - 25)
        self.oz = 50
        self.lx = 25
        self.ly = 25
        self.lz = 45

        # CYLINDER SHAPES
        for i in range(len(self.cylinder_shapes)):
            self.cylinder_figures.append([])

            z = int(self.cylinder_shapes[i][2]) * self.lz - self.lz * 5
            sxy = 20
            sz = 20
            p = []
            self.cylinder_shapes[i][0]['midpts'] = [0,0,z]

            # Get every point
            for j in [-1,1]:
                for k in range(10):
                    rad = (k * 36 / 180) * math.pi
                    p.append( [sxy*cos(rad), sxy*sin(rad), z+j*sz] )

            # Get face of each side
            bot_side = p[:10]
            top_side = p[10:]
            self.cylinder_shapes[i][0]['3dpts'] = [ top_side, bot_side ]
            for l in range(9):
                self.cylinder_shapes[i][0]['3dpts'].append( [p[l], p[l+1], p[l + 11], p[l+10]] )
            self.cylinder_shapes[i][0]['3dpts'].append([p[9], p[0], p[10], p[19]])

            # Project to XY-plane
            for face in self.cylinder_shapes[i][0]['3dpts']:
                vtx2d = [ (vtx3d[0]+self.ox, vtx3d[1]+self.oy) for vtx3d in face]
                self.cylinder_shapes[i][0]['2dpts'].append( vtx2d )

        # BOX SHAPES
        for i in range(len(self.box_shapes)):
            self.box_figures.append([])

            rad = (int(self.box_shapes[i][3]) * 18 / 180) * math.pi
            x = int(self.box_shapes[i][4]) * math.cos(rad) * self.lx
            y = int(self.box_shapes[i][4]) * math.sin(rad) * self.ly
            z = int(self.box_shapes[i][2]) * self.lz - self.lz * 5
            sx = 20
            sy = 5
            sz = 20
            self.box_shapes[i][0]['midpts'] = [x,y,z]


            # Get every point
            p1 = [rotate_x(x, x - sx, y, y - sy, rad), rotate_y(x, x - sx, y, y - sy, rad), z - sz]
            p2 = [rotate_x(x, x - sx, y, y - sy, rad), rotate_y(x, x - sx, y, y - sy, rad), z + sz]
            p3 = [rotate_x(x, x - sx, y, y + sy, rad), rotate_y(x, x - sx, y, y + sy, rad), z - sz]
            p4 = [rotate_x(x, x - sx, y, y + sy, rad), rotate_y(x, x - sx, y, y + sy, rad), z + sz]
            p5 = [rotate_x(x, x + sx, y, y - sy, rad), rotate_y(x, x + sx, y, y - sy, rad), z - sz]
            p6 = [rotate_x(x, x + sx, y, y - sy, rad), rotate_y(x, x + sx, y, y - sy, rad), z + sz]
            p7 = [rotate_x(x, x + sx, y, y + sy, rad), rotate_y(x, x + sx, y, y + sy, rad), z - sz]
            p8 = [rotate_x(x, x + sx, y, y + sy, rad), rotate_y(x, x + sx, y, y + sy, rad), z + sz]

            # Get face of each side
            self.box_shapes[i][0]['3dpts'] = [[p1, p2, p4, p3], [p5, p6, p8, p7], [p1, p2, p6, p5],
                                              [p3, p4, p8, p7],
                                              [p2, p4, p8, p6], [p1, p3, p7, p5]]

            # Project to XY-plane
            for face in self.box_shapes[i][0]['3dpts']:
                vtx2d = [ (vtx3d[0]+self.ox, vtx3d[1]+self.oy) for vtx3d in face]
                self.box_shapes[i][0]['2dpts'].append(vtx2d)


        # Draw the cylinders and boxes
        self.draw_3d_objects()

        # Activate bind
        self.setup_highlight_object_mode = False
        self.active_highlight_object_mode = False
        self.highlight_page_mode = False
        self.view_2d_mode = False
        self.highlight_2D_color = False
        self.three_d_model_canvas.bind('<Button-1>', self.get_mouse_posn)
        self.three_d_model_canvas.bind('<B1-Motion>', self.update_shapes)

    def draw_3d_objects(self):
        self.cylinder_figures = []
        self.box_figures = []

        for i in range(len(self.cylinder_shapes)):
            self.cylinder_figures.append([])

            col = '#%02x%02x%02x' % (self.cylinder_shapes[i][1][0],
                                     self.cylinder_shapes[i][1][1], self.cylinder_shapes[i][1][2])
            for face_idx, face_2d in enumerate(self.cylinder_shapes[i][0]['2dpts']):
                coord_face = ()
                for pt_2d in face_2d:
                    coord_face += pt_2d
                if len(face_2d) == 10:
                    self.cylinder_figures[i].append(
                        self.three_d_model_canvas.create_polygon(coord_face, outline="", fill=col))
                else:
                    self.cylinder_figures[i].append(
                        self.three_d_model_canvas.create_polygon(coord_face, outline="", fill=col))
                self.cylinder_shapes[i][0]['figure_IDs'].append(self.cylinder_figures[i][face_idx])
                self.three_d_model_canvas.tag_bind(self.cylinder_figures[i][face_idx], '<Button-1>',
                                                   self.setup_highlight_object)
                self.three_d_model_canvas.tag_bind(self.cylinder_figures[i][face_idx], '<Button-1>',
                                                   self.setup_highlight_object)
                self.three_d_model_canvas.tag_bind(self.cylinder_figures[i][face_idx], '<B1-Motion>',
                                                   self.cancel_highlight_object)
                self.three_d_model_canvas.tag_bind(self.cylinder_figures[i][face_idx], '<ButtonRelease-1>',
                                                   lambda event, idx=i, type='cylinder': self.highlight_object(
                                                       event, idx, type))

        for i in range(len(self.box_shapes)):
            self.box_figures.append([])
            col = '#%02x%02x%02x' % (self.box_shapes[i][1][0],
                                     self.box_shapes[i][1][1], self.box_shapes[i][1][2])
            self.page_ID = self.box_shapes[i][0]['page_ID']
            for face_idx, face_2d in enumerate(self.box_shapes[i][0]['2dpts']):
                coord_face = ()
                for pt_2d in face_2d:
                    coord_face += pt_2d
                self.box_figures[i].append(
                    self.three_d_model_canvas.create_polygon(coord_face, outline="", fill=col))
                self.box_shapes[i][0]['figure_IDs'].append(self.box_figures[i][face_idx])
                self.three_d_model_canvas.tag_bind(self.box_figures[i][face_idx], '<Button-2>',
                                                   lambda event, key=self.page_ID: self.highlight_page(event, key))
                self.three_d_model_canvas.tag_bind(self.box_figures[i][face_idx], '<Button-1>',
                                                   self.setup_highlight_object)
                self.three_d_model_canvas.tag_bind(self.box_figures[i][face_idx], '<B1-Motion>',
                                                   self.cancel_highlight_object)
                self.three_d_model_canvas.tag_bind(self.box_figures[i][face_idx], '<ButtonRelease-1>',
                                                   lambda event, idx=i, type='box': self.highlight_object(event, idx, type))
        self.update_stacking_order_of_objects()

    def setup_highlight_object(self, event):
        self.setup_highlight_object_mode = True

    def cancel_highlight_object(self, event):
        if self.setup_highlight_object_mode == True:
            self.setup_highlight_object_mode = False

    def highlight_object(self, event, idx, type):
        if self.setup_highlight_object_mode == True:
            selected_hvc_color = None
            if self.highlight_mode == False:
                self.last_highlighted_2D_col = idx
                self.active_idx_and_type = [idx, type]
                if type=='box':
                    rgb_of_hvc = self.box_shapes[idx][1]
                    hvc = '({},{},{})'.format(self.box_shapes[idx][5], self.box_shapes[idx][2], self.box_shapes[idx][4])
                    selected_hvc_color = {'hvc':hvc, 'rgb_of_hvc':rgb_of_hvc}
                    for face_id in self.box_shapes[idx][0]['figure_IDs']:
                        self.three_d_model_canvas.itemconfig(face_id, width=2, outline='black')
                if type == 'cylinder':
                    rgb_of_hvc = self.cylinder_shapes[idx][1]
                    hvc = '({},{},{})'.format(self.cylinder_shapes[idx][4], self.cylinder_shapes[idx][2],
                                              self.cylinder_shapes[idx][3])
                    selected_hvc_color = {'hvc': hvc, 'rgb_of_hvc': rgb_of_hvc}
                    for face_id in self.cylinder_shapes[idx][0]['figure_IDs']:
                        self.three_d_model_canvas.itemconfig(face_id, width=2, outline='black')
                print('begin highlight image')
                self.update_stacking_order_of_objects()
                self.update_HVC_info(selected_hvc_color)

            elif self.highlight_mode == True:
                self.highlight_mode = False
                self.unhighlight_pixels()

    def highlight_2D_view(self, event, idx, hvc):
        if self.highlight_mode == False:
            self.two_d_model_canvas.itemconfig(idx, width=3, outline='black')
            self.update_HVC_info(hvc)
            self.last_highlighted_2D_col = idx
        else:
            self.highlight_mode = False
            self.two_d_model_canvas.itemconfig(self.last_highlighted_2D_col, width=1, outline='black')
            self.unhighlight_pixels()

    def highlight_page(self, event, key):
        if self.highlight_page_mode == False:
            self.highlight_page_mode = True
            self.highlighted_box_objects = [box for box in self.box_shapes if box[0]['page_ID']==key]
            self.active_2D_page_hue = key
            for obj in self.highlighted_box_objects:
                for face_id in obj[0]['figure_IDs']:
                    self.three_d_model_canvas.itemconfig(face_id, width=1, outline='black')
            for obj in self.cylinder_shapes:
                for face_id in obj[0]['figure_IDs']:
                    self.three_d_model_canvas.itemconfig(face_id, width=1, outline='black')
            self.update_stacking_order_of_objects()
            self.view_2d_page_button['state'] = 'normal'
        else:
            self.highlight_page_mode = False
            for obj in self.highlighted_box_objects:
                for face_id in obj[0]['figure_IDs']:
                    self.three_d_model_canvas.itemconfig(face_id, outline='')
            for obj in self.cylinder_shapes:
                for face_id in obj[0]['figure_IDs']:
                    self.three_d_model_canvas.itemconfig(face_id, outline='')
            self.view_2d_page_button['state'] = 'disabled'

    def view_2D(self):
        if self.view_2d_mode == False:
            # Clear canvas
            self.two_d_model_canvas.delete('all')

            # Lifting 2D canvas and hiding 3D canvas
            self.view_2d_mode = True
            self.view_2d_page_button.config(text='View 3D model')
            self.three_d_model_canvas.grid_forget()
            self.two_d_model_canvas.grid(row=0, column=1)

            w_2D = 20
            h_2D = 20
            start_y = int(self.three_d_model_canvas.winfo_height()*0.8)

            hues = {1:'', -1:''}
            # Draw 2D page
            for obj in self.highlighted_box_objects:
                direction = 1 if obj[3] < 10 else -1
                hues[direction] = obj[5]
                x = int(obj[4]) * self.lx * direction
                y = int(obj[2]) * self.lz
                col = '#%02x%02x%02x' % (obj[1][0], obj[1][1], obj[1][2])
                tmp = self.two_d_model_canvas.create_rectangle(x-w_2D+self.ox, start_y-y+h_2D,
                                                         x+w_2D+self.ox, start_y-y-h_2D, fill=col)

                rgb_of_hvc = obj[1]
                hvc = '({},{},{})'.format(obj[5], obj[2], obj[4])
                selected_hvc_color = {'hvc': hvc, 'rgb_of_hvc': rgb_of_hvc}
                self.two_d_model_canvas.tag_bind(tmp, '<Button-1>',
                               lambda event, idx=tmp, hvc=selected_hvc_color : self.highlight_2D_view(event, idx, hvc))

            for obj in self.cylinder_shapes:
                y = int(obj[2]) * self.lz
                col = '#%02x%02x%02x' % (obj[1][0], obj[1][1], obj[1][2])
                tmp = self.two_d_model_canvas.create_rectangle(-1*w_2D+self.ox, start_y-y+h_2D,
                                                         w_2D+self.ox, start_y-y-h_2D, fill=col)

                rgb_of_hvc = obj[1]
                hvc = '({},{},{})'.format(obj[4], obj[2], obj[3])
                selected_hvc_color = {'hvc': hvc, 'rgb_of_hvc': rgb_of_hvc}
                self.two_d_model_canvas.tag_bind(tmp, '<Button-1>',
                                lambda event, idx=tmp, hvc=selected_hvc_color: self.highlight_2D_view(event, idx, hvc))


            # Write Page Label
            self.page_label_text = self.two_d_model_canvas.create_text(self.ox, 100, font="Times 30 bold",
                                   text="HUE: {} - {}".format(hues[-1], hues[1]))

        else:
            # Lifting 3D canvas and hiding 2D canvas
            self.view_2d_mode = False
            self.two_d_model_canvas.grid_forget()
            self.three_d_model_canvas.grid(row=0, column=1)
            self.view_2d_page_button.config(text='View Selected Page')



    def update_shapes(self, event):
        self.botx, self.boty = event.x, event.y

        # Get rotation matrix
        P1 = [0, 0, self.cam_distance]
        P2 = [self.botx-self.topx, self.topy - self.boty, self.cam_distance]
        rx, ry = get_rotation_matrix_x_y(P1, P2)

        # Rotate X and Y axis for each shape
        # Cylinder Shapes

        for i in range(len(self.cylinder_shapes)):
            # Get new position after rotation
            for tmp, face_fig in enumerate(self.cylinder_shapes[i][0]['3dpts']):
                for tmp_pts, (x,y,z) in enumerate(face_fig):
                    pts = [x,y,z]
                    new_pts = three_d_rotation(pts,ry)
                    new_pts = three_d_rotation(new_pts,rx)
                    self.cylinder_shapes[i][0]['3dpts'][tmp][tmp_pts] = new_pts
            midpts = self.cylinder_shapes[i][0]['midpts']
            midpts = three_d_rotation(midpts,ry)
            midpts = three_d_rotation(midpts,rx)
            self.cylinder_shapes[i][0]['midpts'] = midpts


            # Project to XY-plane
            for tmp, face in enumerate(self.cylinder_shapes[i][0]['3dpts']):
                vtx2d = [ (vtx3d[0]+self.ox, vtx3d[1]+self.oy) for vtx3d in face]
                self.cylinder_shapes[i][0]['2dpts'][tmp] = vtx2d

            # Move the object based on new XY-coordinate
            for tmp, face_2d in enumerate(self.cylinder_shapes[i][0]['2dpts']):
                coord_face = ()
                for pt_2d in face_2d:
                    coord_face += pt_2d
                self.three_d_model_canvas.coords( self.cylinder_figures[i][tmp], coord_face)

        # Box Shapes
        for i in range(len(self.box_shapes)):
            # Get new position after rotation
            for tmp, face_fig in enumerate(self.box_shapes[i][0]['3dpts']):
                for tmp_pts, (x,y,z) in enumerate(face_fig):
                    pts = [x,y,z]
                    new_pts = three_d_rotation(pts,ry)
                    new_pts = three_d_rotation(new_pts,rx)
                    self.box_shapes[i][0]['3dpts'][tmp][tmp_pts] = new_pts
            midpts = self.box_shapes[i][0]['midpts']
            midpts = three_d_rotation(midpts, ry)
            midpts = three_d_rotation(midpts, rx)
            self.box_shapes[i][0]['midpts'] = midpts

            # Project to XY-plane
            for tmp, face in enumerate(self.box_shapes[i][0]['3dpts']):
                vtx2d = [ (vtx3d[0]+self.ox, vtx3d[1]+self.oy) for vtx3d in face]
                self.box_shapes[i][0]['2dpts'][tmp] = vtx2d

            # Move the object based on new XY-coordinate
            for tmp, face_2d in enumerate(self.box_shapes[i][0]['2dpts']):
                coord_face = ()
                for pt_2d in face_2d:
                    coord_face += pt_2d
                self.three_d_model_canvas.coords( self.box_figures[i][tmp], coord_face)

        self.topx, self.topy = event.x, event.y

        self.update_stacking_order_of_objects()

    def update_stacking_order_of_objects(self):
        face_id_and_z = []
        for box in self.box_shapes:
            for face_id, vtx3d in zip(box[0]['figure_IDs'], box[0]['3dpts']):
                z = max(vtx3d, key=itemgetter(2))[2]
                face_id_and_z.append([face_id, z])
        for cyl in self.cylinder_shapes:
            for face_id, vtx3d in zip(cyl[0]['figure_IDs'], cyl[0]['3dpts']):
                z = max(vtx3d, key=itemgetter(2))[2]
                face_id_and_z.append([face_id, z])

        # Sorting from lowest to highest
        face_id_and_z = sorted(face_id_and_z, key=lambda x: x[1])
        for face_id, z in face_id_and_z:
            self.three_d_model_canvas.lift(face_id)

    def change_cursor(self, event):
        self.image_canvas.config(cursor='draft_large')

    def revert_cursor(self, event):
        self.image_canvas.config(cursor='')

    def update_buttons(self):
        self.create_mapping_button()
        self.create_upload_button('Re-Upload Image', 17)
        self.create_crop_button()
        self.create_vectorization_level_menus()

    def highlight_pixels(self, pixel_pos):
        self.vectorized_raw_backup = self.vectorized_raw.copy()
        pixels = self.vectorized_raw.load()
        for i,j in pixel_pos:
            pixels[i,j] = (abs(int(pixels[i,j][0]-255)), abs(int(pixels[i,j][1]-255)), abs(int(pixels[i,j][2]-255)))

        self.vectorized_im = ImageTk.PhotoImage(self.vectorized_raw)
        self.image_canvas.itemconfig(self.vectorized_img, image=self.vectorized_im)

    def unhighlight_pixels(self):
        self.vectorized_raw = self.vectorized_raw_backup
        self.vectorized_im = ImageTk.PhotoImage(self.vectorized_raw)
        self.image_canvas.itemconfig(self.vectorized_img, image=self.vectorized_im)

        try:
            if self.active_idx_and_type[1] == 'box':
                for face_id in self.box_shapes[self.active_idx_and_type[0]][0]['figure_IDs']:
                    self.three_d_model_canvas.itemconfig(face_id, outline='')
            if self.active_idx_and_type[1] == 'cylinder':
                for face_id in self.cylinder_shapes[self.active_idx_and_type[0]][0]['figure_IDs']:
                    self.three_d_model_canvas.itemconfig(face_id, outline='')
        except AttributeError:
            pass

    def update_RGB_info_from_image_selection(self, event):
        # Get selected image pixel
        canv_x, canv_y = event.x, event.y
        w, h = self.vectorized_raw.size
        x = canv_x - int((IMG_WIDTH_MAX - w) / 2)
        y = canv_y - int((IMG_HEIGHT_MAX - h) / 2)
        if x >= w:
            x = w - 1
        if y >= h:
            y = h - 1

        pixels = self.vectorized_raw.load()
        selected_rgb_pixel = pixels[x,y]
        self.update_RGB_info(selected_rgb_pixel)

    def update_HVC_info(self, selected_hvc_color):
        if self.highlight_mode == False:
            self.highlight_mode = True

            w, h = self.vectorized_raw.size
            pixels = self.vectorized_raw.load()

            # Calculate total pixels and set up for pixels highlight
            list_of_rgb_of_selected_hvc = [ rgb for rgb,hvc in self.rgb_hvc_list if hvc==selected_hvc_color['hvc'] ]

            pixel_pos = []
            for rgb in list_of_rgb_of_selected_hvc:
                selected_rgb = (int(rgb[0]),int(rgb[1]),int(rgb[2]) )
                tmp_pos = [ (i,j) for i in range(w) for j in range(h) if pixels[i,j]==selected_rgb]
                pixel_pos += tmp_pos

            rgb_pixel_total = len(pixel_pos)
            rgb_percentage = rgb_pixel_total/(w*h)*100
            self.highlight_pixels(pixel_pos)

            # HVC info
            text = 'HVC {0}'.format(selected_hvc_color['hvc'])
            try:
                self.text_RGB.destroy()
            except:
                pass
            self.text_RGB = tk.Label(self.pixels_info_frame, text=text,
                                     font=("Arial", 14))
            self.text_RGB.pack(side='top', padx=5, anchor='nw')

            # Percentage info
            text = 'Percentage {:.3f}%'.format(rgb_percentage)

            try:
                self.text_RGB_percentage.destroy()
            except:
                pass
            self.text_RGB_percentage = tk.Label(self.pixels_info_frame, text=text,
                                     font=("Arial", 14))
            self.text_RGB_percentage.pack(side='top', padx=5, anchor='nw')

            # Color Plate
            try:
                self.color_board.destroy()
            except:
                pass
            col = '#%02x%02x%02x' % selected_hvc_color['rgb_of_hvc']
            self.color_board = tk.Label(self.pixels_info_frame,
                                         background=col, height=5, width=15,)
            self.color_board.pack(side='top', padx=5, anchor='nw')
        else:
            self.highlight_mode = False
            self.unhighlight_pixels()

    def update_RGB_info(self, selected_rgb_pixel):
        if self.highlight_mode == False:
            self.highlight_mode = True

            w, h = self.vectorized_raw.size
            pixels = self.vectorized_raw.load()

            # Calculate total pixels and set up for pixels highlight
            pixel_pos= [ (i,j) for i in range(w) for j in range(h) if pixels[i,j]==selected_rgb_pixel]
            rgb_pixel_total = len(pixel_pos)
            print(selected_rgb_pixel, type(selected_rgb_pixel), w,h)

            rgb_percentage = rgb_pixel_total/(w*h)*100
            self.highlight_pixels(pixel_pos)

            # RGB info
            text = 'RGB {0}'.format(selected_rgb_pixel)
            try:
                self.text_RGB.destroy()
            except:
                pass
            self.text_RGB = tk.Label(self.pixels_info_frame, text=text,
                                     font=("Arial", 14))
            self.text_RGB.pack(side='top', padx=5, anchor='nw')

            # Percentage info
            text = 'Percentage {:.1f}%'.format(rgb_percentage)

            try:
                self.text_RGB_percentage.destroy()
            except:
                pass
            self.text_RGB_percentage = tk.Label(self.pixels_info_frame, text=text,
                                     font=("Arial", 14))
            self.text_RGB_percentage.pack(side='top', padx=5, anchor='nw')

            # Color Plate
            try:
                self.color_board.destroy()
            except:
                pass
            col = '#%02x%02x%02x' % selected_rgb_pixel
            self.color_board = tk.Label(self.pixels_info_frame,
                                         background=col, height=5, width=15,)
            self.color_board.pack(side='top', padx=5, anchor='nw')
        else:
            self.highlight_mode = False
            self.unhighlight_pixels()

    def get_mouse_posn(self, event):
        self.topx, self.topy = event.x, event.y

    def update_sel_rect(self, event):
        self.botx, self.boty = event.x, event.y
        if self.botx < self.topx:
            self.botx,self.topx = self.topx,self.botx
        if self.boty < self.topy:
            self.boty,self.topy = self.topy,self.boty
        self.canv.coords(self.crop_rect, self.topx, self.topy, self.botx, self.boty)  # Update selection rect.

    def execute_crop(self, event):
        w,h = self.im.size

        self.cropped_im = self.im.crop((self.topx-int((self.canv_w-w)/2), self.topy, self.botx-int((self.canv_w-w)/2), self.boty))
        self.cropped_image = ImageTk.PhotoImage(self.cropped_im)

        self.canv.delete('all')
        self.imgtag = self.canv.create_image(self.topx, self.topy, anchor="nw", image=self.cropped_image)

        self.button_crop.destroy()
        self.button_crop = tk.Button(self.button_frame, text='Undo Crop',
                                     font=("Courier", 13), command=lambda: [self.undo_crop()])
        self.button_crop.config(height=3, width=12)
        self.button_crop.pack(side='left')

        self.canv.unbind('<Button-1>')
        self.canv.unbind('<B1-Motion>')
        self.canv.unbind('<ButtonRelease-1>')

        self.use_cropped_image = True

    def undo_crop(self):
        self.canv.delete('all')
        self.imgtag = self.canv.create_image(self.canv_w/2, 0, anchor="n", image=self.img)
        self.create_crop_button()
        self.use_cropped_image = False


    def crop(self):
        self.button_crop.config(height = 3, width = 9, font=("Courier", 14,'bold'))

        self.canv.bind('<Button-1>', self.get_mouse_posn)
        self.canv.bind('<B1-Motion>', self.update_sel_rect)
        self.canv.bind('<ButtonRelease-1>', self.execute_crop)

        # Create selection rectangle (invisible since corner points are equal).
        self.topx, self.topy, self.botx, self.boty = 0, 0, 0, 0
        self.crop_rect = self.canv.create_rectangle(self.topx, self.topy, self.botx, self.boty,
                                          fill='', outline='black')

class SavedImagesPage(Page):
   def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        label = tk.Label(self, text="This is page 2")
        label.pack(side="top", fill="both", expand=True)

class GalleryPage(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        label = tk.Label(self, text="This is page 3")
        label.pack(side="top", fill="both", expand=True)


class App(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        self.master = master
        # self.canvas = tk.Canvas(self.master, bg="grey", width=700, height=500)
        # self.canvas.pack(side=tk.TOP)

        tk.Frame.__init__(self, *args, **kwargs)
        self.p1 = UploadImagePage(self, self.master)
        self.p2 = SavedImagesPage(self)
        self.p3 = GalleryPage(self)

        self.buttonframe = tk.Frame(self)
        self.container = tk.Frame(self)
        self.buttonframe.pack(side="top", fill="x", expand=False)
        self.container.pack(side="top", fill="both", expand=True)

        self.p1.place(in_=self.container, x=0, y=0, relwidth=1, relheight=1)
        self.p2.place(in_=self.container, x=0, y=0, relwidth=1, relheight=1)
        self.p3.place(in_=self.container, x=0, y=0, relwidth=1, relheight=1)

        self.b1 = tk.Button(self.buttonframe, text="Image Mapping", font=("Arial", 16, 'bold'), height = 2, width = 15, command=lambda:[self.p1.lift(),self.unlift_buttons(),self.p1.active(self.b1)])
        self.b2 = tk.Button(self.buttonframe, text="Saved Images", height = 2, width = 15, command=lambda:[self.p2.lift(),self.unlift_buttons(),self.p2.active(self.b2)])
        self.b3 = tk.Button(self.buttonframe, text="Gallery", height = 2, width = 15, command=lambda:[self.p3.lift(),self.unlift_buttons(),self.p3.active(self.b3)])

        self.b1.pack(side="left")
        self.b2.pack(side="left")
        self.b3.pack(side="left")

        self.p1.show()
        self.master.after(0, self.animation)

    def unlift_buttons(self):
        self.b1.config(font=("Arial", 14))
        self.b2.config(font=("Arial", 14))
        self.b3.config(font=("Arial", 14))

    def animation(self):
        self.update_graph()
        # self.master.after(50, self.animation)

    def update_graph(self):
        pass

root = tk.Tk()

SCREEN_WIDTH_INIT = int( root.winfo_screenwidth()/2 )
SCREEN_HEIGHT_INIT = int( root.winfo_screenheight()*0.9 )
SCREEN_WIDTH_RESULTS = root.winfo_screenwidth() - 45
SCREEN_HEIGHT_RESULTS = root.winfo_screenheight()

IMG_WIDTH_MAX = SCREEN_WIDTH_INIT - 10
IMG_HEIGHT_MAX = SCREEN_HEIGHT_INIT - 170

root.config (bg = 'white')
root.resizable(False, False)
root.geometry("{}x{}".format( SCREEN_WIDTH_INIT, SCREEN_HEIGHT_INIT ))
app = App(root)
app.pack(side="top", fill="both", expand=True)
root.mainloop()
