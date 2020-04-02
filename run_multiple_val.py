import glob
import os
import pathlib

val_config_folder = 'project_work_dirs/multiple_damage_detection/configs/val_config/'
val_configs = glob.glob(val_config_folder + '*')
dist_test_script = './tools/dist_test.sh'
checkpoint_root = 'project_work_dirs/multiple_damage_detection/work_dirs'
gpu_num = '3'
test_option = '--eval bbox segm'
val_result_folder = 'val_result_pkl'

for val_config in val_configs:
    val_config_base = os.path.basename(val_config)
    val_config_base, _ = os.path.splitext(val_config_base)

    pathlib.Path(os.path.join(val_result_folder, val_config_base)).mkdir(parents=True, exist_ok=True)

    for num in range(31, 51):
        checkpoint = os.path.join(checkpoint_root, val_config_base, 'epoch_' + str(num) + '.pth')

        val_save_dir = os.path.join(val_result_folder, val_config_base, 'epoch_' + str(num) + '_results.pkl')
        result_path = '--out ' + val_save_dir

        os.system(dist_test_script + ' ' + val_config + ' ' + checkpoint + ' ' + gpu_num + ' ' + result_path + ' ' + test_option)



