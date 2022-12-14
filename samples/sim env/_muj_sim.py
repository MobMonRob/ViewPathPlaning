import mujoco
import glfw
import json
import time


def set_joint_config(data_, dest_joint_config_):
    for i in range(0,len(dest_joint_config_)):    
        data_.actuator('ur5:joint'+str((i+1))).ctrl[0]= dest_joint_config_[i] 
    return data_


def init_window(max_width, max_height):
    glfw.init()
    window = glfw.create_window(width=max_width, height=max_height,
                                       title='Demo', monitor=None,
                                       share=None)
    glfw.make_context_current(window)
    return window











window = init_window(1600, 1000)

model = mujoco.MjModel.from_xml_path('../06_mujocoModels/ur5e_neu.xml')

data = mujoco.MjData(model)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

width, height = glfw.get_framebuffer_size(window)
viewport = mujoco.MjrRect(0, 0, width, height)

scene = mujoco.MjvScene(model, 1000)
camera = mujoco.MjvCamera()
camera.type = 2 # "mjCAMERA_FIXED"
camera.fixedcamid = 0

mujoco.mjv_updateScene(
    model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
    camera, mujoco.mjtCatBit.mjCAT_ALL, scene)


# do initial step to initialize model and data
mujoco.mj_step(model, data)
 

joint_config_sequence = []

filename = "00_Output Collection/output 220909-172211.json" 



with open(filename, 'r') as f:
    dataCollection = json.load(f)

def getSeqNumber(elem):
    return elem[1]

rbPositions = dataCollection["robot"]["rbPos_greedy"]
joint_config_sequence_perRbPos = []
for _rbPosKey in list(rbPositions):
    joint_config_sequence = []
    for _vpKey in rbPositions[_rbPosKey]["VPs"]:
        joint_config_sequence.append([rbPositions[_rbPosKey]["VPs"][_vpKey]["RobotJointConfiguration (radian)"], rbPositions[_rbPosKey]["VPs"][_vpKey]["sequence number"]])
    joint_config_sequence.append([dataCollection["robot"]["rb_ini_jnt_config"],-1])
    joint_config_sequence.sort(key=getSeqNumber)
    joint_config_sequence.append([dataCollection["robot"]["rb_ini_jnt_config"],-1])


    joint_config_sequence_perRbPos.append(joint_config_sequence)



sequence_iterator = 0
sim_seq = joint_config_sequence_perRbPos[0]
last_seq= len(sim_seq)-1

ts = time.time()


while(not glfw.window_should_close(window)):
    mujoco.mj_step(model, data)

    # go to next position every 3 seconds
    if time.time()-ts > 2.5 and sequence_iterator <= last_seq:
        data=set_joint_config(data,sim_seq[sequence_iterator][0])
        ts = time.time()
        sequence_iterator += 1

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()