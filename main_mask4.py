from unet import *
from data_inpute_mask import inpute, process, inpute_test
import numpy as np
import tensorflow as tf
import scipy.io as scio
import matplotlib.pyplot as plt
import xlwt
import xlrd
from xlutils.copy import copy
import os

is_train = 1                                                                #训练过程为0   测试过程为1  模拟样本为2 实验过程为3
use_checkpoint = 1                                                            #训练过程是否载入以前的数据
sample_num = 350                                                          # yangbenshu
checkpoint2_dir = '/data1/csj/1216/old/logs_T2_4/progress-%d'
train_step = 1000000                                                           #总迭代次数
acc_view = 100                                                                 #显示步数和训练误差
step_save = 1000                                                               #显示测试误差的迭代数
step_test = 100                                                                #计算测试误差的迭代数
lr = 0.0001
lr_array = [lr, lr/10, lr/100, lr/100]
expand_num = 256

# test_path = '/data2/csj/data_all/complex_k2_brain_FA15_ex'
# test_path = '/data1/csj/data_all/meas_MID00752_FID11291_a_oled_four_ve_fa15_ex'
# test_path = '/data1/csj/code/meas_MID00766_FID11305_a_oled_four_ve_fa15_complex_k2_ex'
test_path = '/data1/csj/meas_MID00770_FID11309_a_oled_four_ve_fa30_complex_k_ex'
# test_path = '/data1/csj/tempd_case3_T1'
# test_path = '/data2/csj/data_all/20180813/111/csj_median/1001/meas_MID00772_FID11311_a_oled_four_ve_fa15_complex_k_ex'
start_step =90                                                                 #实验时起始步数、结束步数、隔多少步显示
over_step = 90
which_step = 10000

color_bar2 = 0.8
color_bar1 = 1.2

loss_mse=0
acc_xl = 0.0                                                                   #训练误差
acc_test = 0.0                                                                 #测试误差
acc_xl1 = 0.0
acc_xl2 = 0.0
k = 0
checkpoint_dir = 'logs_T2_4/'
style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on')
wb = xlwt.Workbook()
ws = wb.add_sheet('wucafenxi_T2_4')

with tf.name_scope('inpute'):
    if is_train == 1:
        X = tf.placeholder("float", [None, None, None, 2])
        Y = tf.placeholder("float", [None, None, None, 1])
        mask = tf.placeholder("float", [None, None, None, 1])
        learning_rate = tf.placeholder("float", [])
    else:
        X = tf.placeholder("float", [None, None, None, 2])
        Y = tf.placeholder("float", [None, None, None, 1])
        mask = tf.placeholder("float", [None, None, None, 1])
        learning_rate = tf.placeholder("float", [])

# ResNet Models
net = model(X)

with tf.name_scope('loss'):

    change1 = Y <= 0.05
    change1 = tf.cast(change1, tf.float32)
    Y1 = change1*0.05

    change2 = Y > 0.05
    change2 = tf.cast(change2, tf.float32)
    Y2 = tf.multiply(Y, change2)

    Y_change = Y1 + Y2

    # mse = tf.reduce_mean(tf.abs(Y - net))

    # mse = tf.reduce_mean(tf.abs((Y - net) / Y_change))

    # mse = tf.reduce_mean(tf.abs((Y - net) / Y_change)) + 0.001*tf.reduce_sum(tf.multiply(tf.image.total_variation(Y - net), mask))

   # mse = tf.reduce_mean(tf.square(Y - net))
    #
    mse1 = tf.reduce_mean(tf.square((Y - net)))
    #
    # mse1 = tf.reduce_mean(tf.square((net)))
    mse2= 0*tf.reduce_sum(tf.image.total_variation(tf.multiply((Y - net), mask)))
    mse = mse1+mse2

    TV = tf.image.total_variation(Y - net)
with tf.name_scope('train'):
#    train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(mse)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(mse)

with tf.name_scope('acc'):
     acc = tf.reduce_mean(tf.square(Y - net))
     psnr= 10.0 * tf.log(1.0 / (acc)) / tf.log(10.0)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存0
sess = tf.Session(config = config)
#sess = tf.Session()
merged = tf.summary.merge_all()
#writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

if is_train == 0:
    X1, Y1, mask1 = inpute('/data1/csj/data_2000/350_m0_noise_6e7_035_16_only_T2', 'train')
    X_train_pre = np.zeros((int(sample_num*0.8), expand_num, expand_num, 2), dtype=np.float32)
    Y_train_pre = np.zeros((int(sample_num*0.8), expand_num, expand_num, 1), dtype=np.float32)
    mask_train_pre = np.zeros((int(sample_num*0.8), expand_num, expand_num, 1), dtype=np.float32)
    X_test_pre = np.zeros((int(sample_num*0.2), expand_num, expand_num, 2), dtype=np.float32)
    Y_test_pre = np.zeros((int(sample_num*0.2), expand_num, expand_num, 1), dtype=np.float32)
    mask_test_pre = np.zeros((int(sample_num*0.2), expand_num, expand_num, 1), dtype=np.float32)
    X_train_pre[:, :, :, :] = X1[0:int(sample_num*0.8), :, :, :]
    Y_train_pre[:, :, :, :] = Y1[0:int(sample_num*0.8), :, :, :]
    mask_train_pre[:, :, :, :] = mask1[0:int(sample_num*0.8), :, :, :]
    X_test_pre[:, :, :, :] = X1[int(sample_num*0.8):sample_num, :, :, :]
    Y_test_pre[:, :, :, :] = Y1[int(sample_num*0.8):sample_num, :, :, :]
    mask_test_pre[:, :, :, :] = mask1[int(sample_num*0.8):sample_num, :, :, :]
    X1 = 0
    Y1 = 0
    mask1 = 0
    if use_checkpoint == 1:
        f = open('logs_T2_4/checkpoint', 'r')
        line = f.readline()
        step = ''
        for i in line:
            if i.isdigit():
                step = step + i
        begin = int(step)
        k = int(begin/step_save)
        f.close()
        print("Restoring from checkpoint_train", checkpoint)
        saver.restore(sess, checkpoint)
        data = xlrd.open_workbook('误差统计_T2_4.xls', formatting_info=True)
        wb =copy(data)
        ws = wb.get_sheet(0)

    else:
        begin = 0
        k = 0
        print("Don't use checkpoint to restore, Starting now.")

    for i in range(begin, train_step):
        X_train, Y_train, mask_train = process(X_train_pre, Y_train_pre, mask_train_pre)

        feed_dict = {
            X: X_train,
            Y: Y_train,
            mask: mask_train,
            learning_rate: lr_array[int(i/(train_step*0.8))]
            }

        acc_xl = sess.run([train_op, acc], feed_dict=feed_dict)[1]
        acc_xl1 += acc_xl
        # print("loss_mse mse %.8f  tv %.8f"  % (loss_mse,loss_tv))
        if (i+1-acc_view) % acc_view == 0:
            acc_xl1 /= acc_view
            print("目前迭代次数：%d  训练误差%f 学习速率%f" % (i+1, acc_xl1, lr_array[int(i/(train_step*0.8))]))
            acc_xl2 = acc_xl1
            acc_xl1 = 0.0
            # print("loss_mse mse %.8f  tv %.8f" % (loss_mse, loss_tv))

        if (i + 1 - step_save) % step_save == 0:
            print("save train data %d" % (i + 1))
            saver.save(sess, checkpoint_dir + "/progress", global_step=i + 1)
            for j in range(step_test):
                X_test, Y_test, mask_test = process(X_test_pre, Y_test_pre, mask_test_pre)
                acc_test += sess.run([acc], feed_dict={
                    X: X_test,
                    Y: Y_test
                 })[0]
            acc_test /= step_test
            print("总共迭代了%d次,测试误差： " % (i+1))
            print(acc_test)
            print("训练误差：")
            print(acc_xl2)
            ws.write(k, 0, acc_xl2)
            ws.write(k, 1, acc_test)
            acc_test = 0.0
            k += 1
            wb.save('误差统计_T2_4.xls')

elif is_train == 1:
     print("Restoring from checkpoint", checkpoint)
     saver.restore(sess, checkpoint)

     for i in range(1):
         X_test, Y_test, mask = inpute('/data1/csj/data_2000/350_m0_noise_6e7_035_16_only_T2', 'train')
         w = 190
         inpute_one = np.zeros((1, expand_num, expand_num, 2), dtype=np.float32)
         outpute_one = np.zeros((1, expand_num, expand_num, 1), dtype=np.float32)
         inpute_one[:, :, :, :] = X_test[w, :, :, :]
         outpute_one[:, :, :, :] = Y_test[w, :, :, :]
         mask[:, :, :, :] = mask[w, :, :, :]
         mask_one=np.zeros((256,256),dtype=np.float32)
         tv=np.zeros((256,256),dtype=np.float32)
         mask_one=mask[0,:,:,0]

         result_T2 = np.zeros((expand_num, expand_num), dtype=np.float32)
         # result_m0 = np.zeros((expand_num, expand_num), dtype=np.float32)
         # result_B1 = np.zeros((expand_num, expand_num), dtype=np.float32)

         test = net
         test, acc_test, tv1 = sess.run([test, acc, TV], feed_dict={
             X: inpute_one,
             Y: outpute_one
             })
         result_T2[:, :] = test[0, :, :, 0]
         # result_m0[:, :] = test[0, :, :, 1]
         # result_B1[:, :] = test[0, :, :, 2]
         # tv = tv1[0, :, :, 0]

     labels_T2 = np.zeros((expand_num, expand_num), dtype=np.float32)
     # labels_m0 = np.zeros((expand_num, expand_num), dtype=np.float32)
     # labels_B1 = np.zeros((expand_num, expand_num), dtype=np.float32)
     labels_T2[:, :] = outpute_one[0, :, :, 0]
     # labels_m0[:, :] = outpute_one[0, :, :, 1]
     # labels_B1[:, :] = outpute_one[0, :, :, 2]
     print("最后得到图像的误差为：")
     print(acc_test)

     plt.figure(1)
     plt.subplot(231)
     plt.imshow((inpute_one[0, :, :, 0]**2+inpute_one[0, :, :, 1]**2)**0.5)
     plt.title('input')

     plt.subplot(234)
     plt.imshow(result_T2,vmin=0, vmax=0.19, cmap = 'jet')
     plt.title('result')

     plt.subplot(235)
     plt.imshow(labels_T2, vmin=0, vmax=0.19, cmap = 'jet')
     plt.title('labels')

     plt.subplot(236)
     plt.imshow(np.abs(result_T2 - labels_T2), vmin=0, vmax=0.19, cmap = 'jet')
     # plt.title('result - label')

     # plt.subplot(437)
     # plt.imshow(result_m0, vmin=0, vmax=0.5, cmap='jet')
     # # plt.title('result_m0')
     #
     # plt.subplot(438)
     # plt.imshow(labels_m0, vmin=0, vmax=0.5, cmap='jet')
     # # plt.title('labels_m0')
     #
     # plt.subplot(439)
     # plt.imshow(np.abs(result_m0 - labels_m0), vmin=0, vmax=0.19, cmap='jet')
     # plt.title('result - label')

     # plt.subplot(4,3,10)
     # plt.imshow(result_B1, vmin=color_bar2, vmax=color_bar1, cmap='jet')
     # # plt.title('result_B1')
     #
     # plt.subplot(4,3,11)
     # plt.imshow(labels_B1, vmin=color_bar2, vmax=color_bar1, cmap='jet')
     # # plt.title('labels_B1')
     #
     # plt.subplot(4,3,12)
     # plt.imshow(np.abs(result_B1 - labels_B1), vmin=0, vmax=0.19, cmap='jet')
     # plt.title('result - label')
     plt.show()

 #    scio.savemat('result/result_simu.mat', {"result": result})
 #    scio.savemat('result/labels_simu.mat', {"labels": labels})

elif is_train == 2:
     print("Restoring from checkpoint", checkpoint)
     saver.restore(sess, checkpoint)

     for i in range(1):
         X_test, Y_test = inpute('/data2/csj/data_all/20180813/tempd_csj_original2', 'test')
         w = 0
         inpute_one = np.zeros((1, expand_num, expand_num, 2), dtype=np.float32)
         outpute_one = np.zeros((1, expand_num, expand_num, 1), dtype=np.float32)
         inpute_one[:, :, :, :] = X_test[w, :, :, :]
         outpute_one[:, :, :, :] = Y_test[w, :, :, :]
         result = np.zeros((expand_num, expand_num), dtype=np.float32)

         test = net
         test, acc_test = sess.run([test, acc], feed_dict={
             X: inpute_one,
             Y: outpute_one
             })
         result[:, :] = test[0, :, :, 0]

     labels = np.zeros((expand_num, expand_num), dtype=np.float32)
     labels[:, :] = outpute_one[0, :, :, 0]
     print("最后得到图像的误差为：")
     print(acc_test)

     plt.figure(1)
     plt.subplot(221)
     plt.imshow(((inpute_one[0, :, :, 0]**2+inpute_one[0, :, :, 1]**2)**0.5))
     plt.title('input')

     plt.subplot(222)
     plt.imshow(result,vmin=0, vmax=color_bar)
     plt.title('result')

     plt.subplot(223)
     plt.imshow(labels, vmin=0, vmax=color_bar)
     plt.title('labels')

     plt.subplot(224)
     plt.imshow(np.abs(result - labels), vmin=0, vmax=color_bar)
     plt.title('result - label')
     # plt.show()
     result = result*1000
     scio.savemat('result_T2_csj/result_simu.mat', {"result": result})
     scio.savemat('result_T2_csj/labels_simu.mat', {"labels": labels})


elif is_train == 3:
  for kk in range(start_step, over_step+1):
     checkpoint = (checkpoint2_dir)%(kk*which_step)
     w_test = 15
     print("Restoring from checkpoint", checkpoint)
     saver.restore(sess, checkpoint)
     X_test = inpute_test(test_path)
     X_test_in = np.zeros((1, expand_num, expand_num, 2), dtype=np.float32)
     X_test_in[:, :, :, ] = X_test[w_test, :, :, :]
     result = np.zeros((expand_num, expand_num, 3), dtype=np.float32)
     result_T2 = np.zeros((expand_num, expand_num), dtype=np.float32)
     result_m0 = np.zeros((expand_num, expand_num), dtype=np.float32)
     result_B1 = np.zeros((expand_num, expand_num), dtype=np.float32)
     test = net
     test = sess.run([test], feed_dict={
         X: X_test_in
     })[0]
     result_T2[:, :] = test[0, :, :, 0]
     result_m0[:, :] = test[0, :, :, 1]
     result_B1[:, :] = test[0, :, :, 2]

     plt.figure()
     plt.subplot(221)
     plt.imshow((X_test_in[0, :, :, 0] ** 2 + X_test_in[0, :, :, 1] ** 2) ** 0.5, vmin=0, vmax=0.19)
     plt.title('input')

     plt.subplot(222)
     plt.imshow(result_T2, vmin=0, vmax=0.19, cmap = 'jet')
     plt.title(('result_%d')%(kk*which_step))
     plt.subplot(223)
     plt.imshow(result_m0, vmin=0, vmax=0.5, cmap='jet')
     plt.subplot(224)
     plt.imshow(result_B1, vmin=0.8, vmax=1.2, cmap='jet')
     plt.show()
     result=result*1000
     result[:, :, 0] = result_T2
     result[:, :, 1] = result_m0
     result[:, :, 2] = result_B1
     scio.savemat(('rennao_30d_2/result_T2_m0_B1_15.mat'), {"result": result})

elif is_train == 4:
    checkpoint = (checkpoint2_dir) % ((over_step) * which_step)
    print("Restoring from checkpoint", checkpoint)
    saver.restore(sess, checkpoint)
    X_test = inpute_test(test_path)
    X_test_in = np.zeros((1, expand_num, expand_num, 2), dtype=np.float32)
    for k in range(X_test.shape[0]):
       X_test_in[:, :, :, ] = X_test[k, :, :, :]
       result = np.zeros((expand_num, expand_num), dtype=np.float32)
       test = net
       test = sess.run([test], feed_dict={
           X: X_test_in
       })[0]
       result[:, :] = test[0, :, :, 0]
       result = result * 1000
       scio.savemat(('result_0766/result_0766_%d.mat') % (k+1), {"result": result})
sess.close()
#tensorboard --logdir=F:\python\my_demo\logs