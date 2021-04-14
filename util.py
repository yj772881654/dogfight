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
        self.v_x=list[9]
        self.v_z=list[10]
        self.v_y=list[11]
        self.positionX2 = list[12]
        self.positionY2= list[13]
        self.height2 = list[14]
        self.speed2 = list[15]
        self.health2 = list[16]
        self.pinchAtd2 = list[17]
        self.rollAtd2 = list[18]
        self.yaw2=list[19]
        self.target2 = list[20]
        self.v_x2=list[21]
        self.v_z2=list[22]
        self.v_y2=list[23]

def cal_r_height(state:AircraftState):
    #高度奖励
    delta_H=state.height-state.height2
    # if state.height<300:                    #飞行高度低于300惩罚
    #     r_H_self=(state.height)/100-3
    # else:
    #     r_H_self=0

    if delta_H<2000:                        #敌人高度奖励
        r_H=(delta_H+1000)/1000
    else:
        r_H=(2000-delta_H)/1000
    # print("r_H",r_H," r_H_self",r_H_self)
    return r_H

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
def cal_dis(state:AircraftState):
    e=[state.v_x/(state.speed/3.6),state.v_z/(state.speed/3.6),state.v_y/(state.speed/3.6)]
    d=[state.positionX2-state.positionX,state.positionY2-state.positionY,state.height2-state.height]
    e_d=e[0]*d[0]+e[1]*d[1]+e[2]*d[2]
    return e_d
def cal_r_angle(state:AircraftState):
    v_r=[cos(state.yaw)*cos(state.pinchAtd),sin(state.yaw)*cos(state.pinchAtd),sin(state.pinchAtd)]
    v_b=[cos(state.yaw2)*cos(state.pinchAtd2),sin(state.yaw2)*cos(state.pinchAtd2),sin(state.pinchAtd2)]
    # d=[state.positionX-state.positionX2,state.positionY-state.positionY2,state.height-state.height2] #自己的角度
    d = [state.positionX2 - state.positionX, state.positionY2 - state.positionY, state.height2 - state.height]  # 敌人的角度
    #d位置的顺序决定谁的角度
    D=sqrt(pow(state.positionX-state.positionX2,2)+pow(state.positionY-state.positionY2,2)+pow(state.height-state.height2,2))

    AA=0
    ATA=0
    if D!=0:
        AA=acos((v_r[0]*d[0]+v_r[1]*d[1]+v_r[2]*d[2])/D)
        ATA=acos((v_b[0]*d[0]+v_b[1]*d[1]+v_b[2]*d[2])/D)
    return 10-10*(AA+ATA)/pi
def cal_launch_r(state:AircraftState):
    return state.height/20
def action_judge(state1:AircraftState,state2:AircraftState):
    # r_health=-(1-state1.health)
    # print("r_health",r_health)
    if state1.height>5000:
        return -1
    if state1.height>1500:
        if sqrt(pow(state2.positionX-state2.positionX2,2)+pow(state2.positionY-state2.positionY2,2)+pow(state2.height-state2.height,2))>2500:
            return (8000-sqrt(pow(state2.positionX-state2.positionX2,2)+pow(state2.positionY-state2.positionY2,2)+pow(state2.height-state2.height,2)))/2000
        else:
            # print(cal_dis(state2))
            if cal_dis(state2)>0:
                if cal_dis(state2)<500 :
                    return 2
                else:
                    return (2500-cal_dis(state2))/1000
            else:
                return -(2500+cal_dis(state2))/1000
    else:
        # print("launch reward:",cal_launch_r(state2)-cal_launch_r(state1))
        return cal_launch_r(state2)-cal_launch_r(state1)

