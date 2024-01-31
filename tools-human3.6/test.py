a = annotations['ori_img_1000'].detach()
B,S,C,X,Y = a.shape
a = a.reshape(B*S, C, X, Y)
a = a.cpu().numpy()
# a = a.cpu().numpy().transpose(0,2,3,1)

b= pred_3d_vertices_fine.detach().cpu().numpy()
c = pred_cam.detach().cpu().numpy()
color='light_blue'
focal_length=1000
camera = c[0,:]
camera_t = np.array([camera[1], camera[2], 2*focal_length/(1024 * camera[0] +1e-9)])

rend_img = renderer.render(b[0,:,:], camera_t=camera_t,
                            img=a[0,:,:,:]/255, use_bg=True,
                            focal_length=focal_length,
                            body_color=color)

cv2.imwrite(r"/HOME/HOME/Zhongzhangnan/FastMETRO_EMA_adapt/src/tools/test_image.jpg", rend_img*255)