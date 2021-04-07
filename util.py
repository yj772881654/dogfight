from Machines import *
import math
def take_action(action,aircraft:Aircraft):
    if action == 0:     # n_choose
        pass
    elif action == 1:
        aircraft_down(aircraft)
    elif action == 2:
        aircraft_up(aircraft)
    elif action == 3:
        aircraft_left(aircraft)
    elif action == 4:
        aircraft_right(aircraft)
    elif action == 5:
        aircraft_roll_right(aircraft)
    elif action == 6:
        aircraft_roll_left(aircraft)
    elif action == 7:
        aircraft_combution(aircraft)
    # elif action ==8:
    #     aircraft_fire(aircraft)
def aircraft_down(aircraft:Aircraft):
    aircraft.set_pitch_level(1)
def aircraft_up(aircraft:Aircraft):
    aircraft.set_pitch_level(-1)
def aircraft_left(aircraft:Aircraft):
    aircraft.set_roll_level(1)
def aircraft_right(aircraft:Aircraft):
    aircraft.set_roll_level(-1)
def aircraft_roll_right(aircraft:Aircraft):
    aircraft.set_yaw_level(-1)
def aircraft_roll_left(aircraft:Aircraft):
    aircraft.set_yaw_level(1)
def aircraft_combution(aircraft:Aircraft):
    if aircraft.post_combution:
        aircraft.deactivate_post_combution()
    else:
        aircraft.activate_post_combution()
def aircraft_fire(aircraft:Aircraft):
    aircraft.fire_missile()
def angle_trans(x):
    return x/180*pi
def cal_yaw(vx,vz):
	yaw=0
	if vz != 0:
		# yaw=180 / math.pi * math.atan(Main.p1_aircraft.v_move.x / Main.p1_aircraft.v_move.z)
		# print(yaw)
		if vz > 0:
			yaw = 180 / math.pi * math.atan(vx / vz)
		elif vx > 0:
			yaw = 180 / math.pi * math.atan(vx / vz) + 180
		else:
			yaw = 180 / math.pi * math.atan(vx / vz) - 180

	return yaw
class AircraftState:
    def __init__(self,list=[]):
        self.positionX=list[0]
        self.positionY=list[1]
        self.height=list[2]
        self.speed=list[3]
        self.health=list[4]
        self.pinchAtd=list[5]
        self.rollAtd=list[6]
        self.yaw=list[7]
        self.target=list[8]
        self.positionX2 = list[9]
        self.positionY2= list[10]
        self.height2 = list[11]
        self.speed2 = list[12]
        self.health2 = list[13]
        self.pinchAtd2 = list[14]
        self.rollAtd2 = list[15]
        self.yaw2=list[16]
        self.target2 = list[17]

def cal_r_height(state:AircraftState):
    #高度奖励
    delta_H=state.height-state.height2
    if state.height<300:                    #飞行高度低于300惩罚
        r_H_self=(state.height)/100-3
    else:
        r_H_self=0

    if delta_H<2000:                        #敌人高度奖励
        r_H=(delta_H+1000)/1000-1
    else:
        r_H=(2000-delta_H)/1000+2
    # print("r_H",r_H," r_H_self",r_H_self)
    return r_H+r_H_self

def cal_r_v(state:AircraftState):
    r_v=(state.speed-state.speed2)/100
    if state.speed<500:
        r_v_self=state.speed/100-5
    else:
        r_v_self=0
    # print("r_v",r_v," r_v_self",r_v_self)
    return r_v+r_v_self
def angle_trans(x):
    return x/180*pi
def cal_r_angle(state:AircraftState):
    v_r=[cos(state.yaw)*cos(state.pinchAtd),sin(state.yaw)*cos(state.pinchAtd),sin(state.pinchAtd)]
    v_b=[cos(state.yaw2)*cos(state.pinchAtd2),sin(state.yaw2)*cos(state.pinchAtd2),sin(state.pinchAtd2)]
    d=[state.positionX-state.positionX2,state.positionY-state.positionY2,state.height-state.height2]
    #d位置的顺序决定谁的角度
    D=sqrt(pow(state.positionX-state.positionX2,2)+pow(state.positionY-state.positionY2,2)+pow(state.height-state.height2,2))
    AA=0
    ATA=0
    if D!=0:
        AA=acos((v_r[0]*d[0]+v_r[1]*d[1]+v_r[2]*d[2])/D)
        ATA=acos((v_b[0]*d[0]+v_b[1]*d[1]+v_b[2]*d[2])/D)
    return 10-10*(AA+ATA)/pi
def cal_launch_r(state:AircraftState):
    if state.height<1000:
        return state.height/50
    else:
        return state.speed/700+state.height/200
def cal_r(state:AircraftState):

    return cal_r_angle(state)+cal_r_v(state)+cal_r_height(state)
    # return cal_r_v(state)
def action_judge(state1:AircraftState,state2:AircraftState):
    #fire
    # r_health=0
    # if state2.health2-state1.health2<0:
    #     r_health=state1.health2-state2.health2 #命中奖励
    # if state2.health-state1.health<0:
    #     r_health=state2.health-state1.health  #被命中惩罚
    if state1.height>5000:
        return -1

    if state1.height>1500:
        print("reward:",cal_r(state2)-cal_r(state1))
        return cal_r(state2)-cal_r(state1)
    else:
        # print("new:",cal_launch_r(state2))
        # print("old:",cal_launch_r(state1))
        print("launch reward:",cal_launch_r(state2)-cal_launch_r(state1))
        return cal_launch_r(state2)-cal_launch_r(state1)
# new_state = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
# new_state1=[2,3,4,5,8,6,9,9,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
# new=AircraftState(new_state)
# new1=AircraftState(new_state1)
# print(action_judge(new1,new))
