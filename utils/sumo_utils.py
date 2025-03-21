#!/usr/bin/env python
import os
import sys

import csv
import numpy as np
from decimal import Decimal


def generate_routefile(args, save_dir="./sumo_data"):
    step_length = args.step_length  # the sim interval length
    num_steps = args.num_steps  # number of time steps
    Lambda = args.Lambda  # arrival rate of car flow
    accel = args.accel  # accelerate of car flow
    decel = args.decel  # decelerate of car flow
    sigma = (
        args.sigma
    )  # imperfection of drivers, which takes value on [0,1], with 0 meaning perfection and 1 meaning imperfection
    carLength = args.carLength  # length of cars
    minGap = args.minGap  # minimum interval between adjacent cars
    maxSpeed = args.maxSpeed  # maxSpeed for cars

    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, f"road_Lbd{args.Lambda:.2f}.rou.xml"), "w"
    ) as routes:
        print(
            """<?xml version="1.0" encoding="UTF-8"?>

<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="typecar" length="{carLength}" minGap="{minGap}" maxSpeed="{maxSpeed}" accel="{accel}" decel="{decel}" sigma="{sigma}"/>
    <!-- Routes -->
    <route id="r_0" edges="E0 -E2"/>
    <route id="r_1" edges="E0 E1 E5"/>
    <route id="r_10" edges="E2 E1 E6 E9"/>
    <route id="r_11" edges="E2 E3 E10"/>
    <route id="r_12" edges="E2 E3 E11"/>
    <route id="r_13" edges="E2 -E0"/>
    <route id="r_14" edges="-E5 E4"/>
    <route id="r_15" edges="-E5 E6 E8"/>
    <route id="r_16" edges="-E5 E6 E9"/>
    <route id="r_17" edges="-E5 E6 E7 E10"/>
    <route id="r_18" edges="-E5 E6 E7 E11"/>
    <route id="r_19" edges="-E5 -E1 -E0"/>
    <route id="r_2" edges="E0 E1 E4"/>
    <route id="r_20" edges="-E5 -E1 -E2"/>
    <route id="r_21" edges="-E4 E6 E8"/>
    <route id="r_22" edges="-E4 E6 E9"/>
    <route id="r_23" edges="-E4 E6 E7 E10"/>
    <route id="r_24" edges="-E4 E6 E7 E11"/>
    <route id="r_25" edges="-E4 -E1 -E0"/>
    <route id="r_26" edges="-E4 -E1 -E2"/>
    <route id="r_27" edges="-E4 E5"/>
    <route id="r_28" edges="-E8 E9"/>
    <route id="r_29" edges="-E8 E7 E10"/>
    <route id="r_3" edges="E0 E1 E6 E8"/>
    <route id="r_30" edges="-E8 E7 E11"/>
    <route id="r_31" edges="-E8 E7 -E3 -E0"/>
    <route id="r_32" edges="-E8 E7 -E3 -E2"/>
    <route id="r_33" edges="-E8 -E6 E5"/>
    <route id="r_34" edges="-E8 -E6 E4"/>
    <route id="r_35" edges="-E9 E7 E10"/>
    <route id="r_36" edges="-E9 E7 E11"/>
    <route id="r_37" edges="-E9 E7 -E3 -E0"/>
    <route id="r_38" edges="-E9 E7 -E3 -E2"/>
    <route id="r_39" edges="-E9 -E6 E5"/>
    <route id="r_4" edges="E0 E1 E6 E9"/>
    <route id="r_40" edges="-E9 -E6 E4"/>
    <route id="r_41" edges="-E9 E8"/>
    <route id="r_42" edges="-E10 E11"/>
    <route id="r_43" edges="-E10 -E3 -E0"/>
    <route id="r_44" edges="-E10 -E3 -E2"/>
    <route id="r_45" edges="-E10 -E3 E1 E5"/>
    <route id="r_46" edges="-E10 -E3 E1 E4"/>
    <route id="r_47" edges="-E10 -E7 E8"/>
    <route id="r_48" edges="-E10 -E7 E9"/>
    <route id="r_49" edges="-E11 -E3 -E0"/>
    <route id="r_5" edges="E0 E3 E10"/>
    <route id="r_50" edges="-E11 -E3 -E2"/>
    <route id="r_51" edges="-E11 -E3 E1 E5"/>
    <route id="r_52" edges="-E11 -E3 E1 E4"/>
    <route id="r_53" edges="-E11 -E7 E8"/>
    <route id="r_54" edges="-E11 -E7 E9"/>
    <route id="r_55" edges="-E11 E10"/>
    <route id="r_6" edges="E0 E3 E11"/>
    <route id="r_7" edges="E2 E1 E5"/>
    <route id="r_8" edges="E2 E1 E4"/>
    <route id="r_9" edges="E2 E1 E6 E8"/>
    <routeDistribution id="rd_0" routes="r_0 r_1 r_10 r_11 r_12 r_13 r_14 r_15 r_16 r_17 r_18 r_19 r_2 r_20 r_21 r_22 r_23 r_24 r_25 r_26 r_27 r_28 r_29 r_3 r_30 r_31 r_32 r_33 r_34 r_35 r_36 r_37 r_38 r_39 r_4 r_40 r_41 r_42 r_43 r_44 r_45 r_46 r_47 r_48 r_49 r_5 r_50 r_51 r_52 r_53 r_54 r_55 r_6 r_7 r_8 r_9" probabilities="0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50"/>
    <routeDistribution id="rd_1" routes="r_0 r_1 r_2 r_3 r_4 r_5 r_6" probabilities="0.50 0.50 0.50 0.50 0.50 0.50 0.50"/>
    <routeDistribution id="rd_2" routes="r_7 r_8 r_9 r_10 r_11 r_12 r_13" probabilities="0.50 0.50 0.50 0.50 0.50 0.50 0.50"/>
    <routeDistribution id="rd_3" routes="r_14 r_15 r_16 r_17 r_18 r_19 r_20" probabilities="0.50 0.50 0.50 0.50 0.50 0.50 0.50"/>
    <routeDistribution id="rd_4" routes="r_21 r_22 r_23 r_24 r_25 r_26 r_27" probabilities="0.50 0.50 0.50 0.50 0.50 0.50 0.50"/>
    <routeDistribution id="rd_5" routes="r_28 r_29 r_30 r_31 r_32 r_33 r_34" probabilities="0.50 0.50 0.50 0.50 0.50 0.50 0.50"/>
    <routeDistribution id="rd_6" routes="r_35 r_36 r_37 r_38 r_39 r_40 r_41" probabilities="0.50 0.50 0.50 0.50 0.50 0.50 0.50"/>
    <routeDistribution id="rd_7" routes="r_42 r_43 r_44 r_45 r_46 r_47 r_48" probabilities="0.50 0.50 0.50 0.50 0.50 0.50 0.50"/>
    <routeDistribution id="rd_8" routes="r_49 r_50 r_51 r_52 r_53 r_54 r_55" probabilities="0.50 0.50 0.50 0.50 0.50 0.50 0.50"/>
    <!-- Vehicles, persons and containers (sorted by depart) -->
    <!-- flow id="flow0" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{probability}" route="rd_0"/ -->
    <flow id="flow1" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{probability}" route="rd_1"/>
    <flow id="flow2" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{probability}" route="rd_2"/>
    <flow id="flow3" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{probability}" route="rd_3"/>
    <flow id="flow4" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{probability}" route="rd_4"/>
    <flow id="flow5" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{probability}" route="rd_5"/>
    <flow id="flow6" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{probability}" route="rd_6"/>
    <flow id="flow7" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{probability}" route="rd_7"/>
    <flow id="flow8" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{probability}" route="rd_8"/>
    
</routes>
""".format(
                **{
                    "accel": accel,
                    "decel": decel,
                    "sigma": sigma,
                    "carLength": carLength,
                    "minGap": minGap,
                    "maxSpeed": maxSpeed,
                    "num_steps": num_steps,
                    "probability": min(Lambda, 1),
                }
            ),
            file=routes,
        )


def generate_cfgfile(args, save_dir="./sumo_data"):
    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, f"road_Lbd{args.Lambda:.2f}.sumocfg"), "w"
    ) as cfgfile:
        print(
            f"""<?xml version="1.0" encoding="UTF-8"?>

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="road.net.xml"/>
        <route-files value="road_Lbd{args.Lambda:.2f}.rou.xml"/>
    </input>

    <time>
        <begin value="0"/>
    </time>

</configuration>
""",
            file=cfgfile,
        )


def sumo_run(args, save_dir="./sumo_data"):
    # we need to import python modules from the $SUMO_HOME/tools directory
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
        import traci  # noqa
        from sumolib import checkBinary  # noqa
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run

    sumoBinary = checkBinary("sumo")

    # first, generate the route file for this simulation

    # generate_routefile(args,save_dir=save_dir)
    # generate_netfile(args,save_dir=save_dir)
    # generate_cfgfile(args,save_dir=save_dir)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start(
        [
            sumoBinary,
            "-c",
            os.path.join(save_dir, f"road_Lbd{args.Lambda:.2f}.sumocfg"),
            "--tripinfo-output",
            os.path.join(save_dir, f"tripinfo_Lbd{args.Lambda:.2f}.xml"),
            "--step-length",
            str(args.step_length),
        ]
    )

    """execute the TraCI control loop"""
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
    traci.close()
    sys.stdout.flush()


def sumo_run_with_trajectoryInfo(args, save_dir="./sumo_data"):
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run

    # we need to import python modules from the $SUMO_HOME/tools directory
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
        import traci  # noqa
        from sumolib import checkBinary  # noqa
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    sumoBinary = checkBinary("sumo")

    # first, generate the route file for this simulation
    generate_routefile(args, save_dir=save_dir)
    # generate_netfile(args,save_dir=save_dir)
    generate_cfgfile(args, save_dir=save_dir)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start(
        [
            sumoBinary,
            "-c",
            os.path.join(save_dir, f"road_Lbd{args.Lambda:.2f}.sumocfg"),
            "--tripinfo-output",
            os.path.join(save_dir, f"tripinfo_Lbd{args.Lambda:.2f}.xml"),
            "--step-length",
            str(args.step_length),
        ]
    )
    """execute the TraCI control loop"""
    step = 0
    w = csv.writer(open(args.trajectoryInfo_path, "w", newline=""))
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        for veh_i, veh in enumerate(traci.vehicle.getIDList()):
            (
                (x, y),
                speed,
                angle,
            ) = [
                f(veh)
                for f in [
                    traci.vehicle.getPosition,
                    traci.vehicle.getSpeed,  # Returns the speed of the named vehicle within the last step [m/s]; error value: -1001
                    traci.vehicle.getAngle,
                ]
            ]
            w.writerow([step, veh, veh_i, x, y, speed, angle])
    traci.close()
    sys.stdout.flush()


def read_tripInfo(tripInfo_path="tripinfo.xml"):
    pass
    car_tripinfo = []
    with open(tripInfo_path, "r") as f:
        tripinfo = f.readlines()
        for line in tripinfo:
            if line.startswith("    <tripinfo"):
                car_info = line[14:-3].split(" ")
                car_dict = dict()
                for item in car_info:
                    key, value = item.split("=")
                    car_dict[key] = value[1:-1]
                car_tripinfo.append(car_dict)
    return car_tripinfo


def read_trajectoryInfo_carindex(args):
    r = csv.reader(open(args.trajectoryInfo_path, "r"))
    car_trajectory = {}
    for row in r:
        [step, veh, veh_i, x, y, speed] = row
        veh_id = float(veh.split("flow")[-1])
        step, x, y, speed = int(step), float(x), float(y), float(speed)
        pos = np.array([x, y])
        timeslot = float(step * Decimal(str(args.step_length)))
        if not veh_id in car_trajectory:
            car_trajectory[veh_id] = []
        car_trajectory[veh_id].append(
            {
                "t": timeslot,
                "pos": pos,
                "v": speed,
            }
        )
    return car_trajectory


def read_trajectoryInfo_carindex_matrix(args):
    r = csv.reader(open(args.trajectoryInfo_path, "r"))
    car_trajectory = {}
    for row in r:
        [step, veh, veh_i, x, y, speed] = row
        veh_id = float(veh.split("flow")[-1])
        step, x, y, speed = int(step), float(x), float(y), float(speed)
        timeslot = float(step * Decimal(str(args.step_length)))
        if not veh_id in car_trajectory:
            car_trajectory[veh_id] = []
        car_trajectory[veh_id].append([timeslot, x, y, speed])
    for k, v in car_trajectory.items():
        car_trajectory[k] = np.array(v)
    return car_trajectory


def read_trajectoryInfo_timeindex(
    args,
    start_time=0,
    end_time=1e7,
    display_intervel=0.1,
):
    reader = csv.reader(open(args.trajectoryInfo_path, "r"))
    timeline_dir = {}
    last_t = 0
    for row in reader:
        [step, veh, veh_i, x, y, speed, angle] = row
        step, x, y, speed, angle = (
            int(step),
            float(x),
            float(y),
            float(speed),
            float(angle),
        )
        timeslot = float(step * Decimal(str(args.step_length)))
        if (
            timeslot >= start_time
            and timeslot <= end_time
            and timeslot - last_t >= display_intervel
        ):
            timeline_dir[timeslot] = {}
            last_t = timeslot

    reader = csv.reader(open(args.trajectoryInfo_path, "r"))
    for row in reader:
        [step, veh, veh_i, x, y, speed, angle] = row
        step, x, y, speed, angle = (
            int(step),
            float(x),
            float(y),
            float(speed),
            float(angle),
        )
        timeslot = float(step * Decimal(str(args.step_length)))
        if timeslot in timeline_dir.keys():
            veh_id = float(veh.split("flow")[-1])
            pos = np.array([x, y])
            timeline_dir[timeslot][veh_id] = {
                "pos": pos,
                "v": speed,
                "angle": angle,
            }

    for timeslot in timeline_dir:
        timeline_dir[timeslot] = dict(sorted(timeline_dir[timeslot].items()))
    return timeline_dir
