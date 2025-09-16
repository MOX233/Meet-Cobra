#!/usr/bin/env python
import os
import sys

import csv
import numpy as np
from decimal import Decimal


def generate_netfile(args, save_dir="./sumo_data"):
    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, f"road.net.xml"), "w"
    ) as netfile:
        print("""<?xml version="1.0" encoding="UTF-8"?>

<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="0.00,0.00" convBoundary="-500.00,-500.00,500.00,500.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>

    <edge id=":NE_Center_0" function="internal">
        <lane id=":NE_Center_0_0" index="0" speed="6.51" length="9.03" shape="242.00,263.60 241.65,261.15 240.60,259.40 238.85,258.35 236.40,258.00"/>
    </edge>
    <edge id=":NE_Center_1" function="internal">
        <lane id=":NE_Center_1_0" index="0" speed="13.89" length="27.20" shape="242.00,263.60 242.00,236.40"/>
        <lane id=":NE_Center_1_1" index="1" speed="13.89" length="27.20" shape="245.20,263.60 245.20,236.40"/>
        <lane id=":NE_Center_1_2" index="2" speed="13.89" length="27.20" shape="248.40,263.60 248.40,236.40"/>
    </edge>
    <edge id=":NE_Center_4" function="internal">
        <lane id=":NE_Center_4_0" index="0" speed="10.36" length="5.01" shape="248.40,263.60 249.11,258.64"/>
    </edge>
    <edge id=":NE_Center_20" function="internal">
        <lane id=":NE_Center_20_0" index="0" speed="10.36" length="19.50" shape="249.11,258.64 249.35,256.95 252.20,252.20 256.95,249.35 263.60,248.40"/>
    </edge>
    <edge id=":NE_Center_5" function="internal">
        <lane id=":NE_Center_5_0" index="0" speed="6.51" length="9.03" shape="263.60,258.00 261.15,258.35 259.40,259.40 258.35,261.15 258.00,263.60"/>
    </edge>
    <edge id=":NE_Center_6" function="internal">
        <lane id=":NE_Center_6_0" index="0" speed="13.89" length="27.20" shape="263.60,258.00 236.40,258.00"/>
        <lane id=":NE_Center_6_1" index="1" speed="13.89" length="27.20" shape="263.60,254.80 236.40,254.80"/>
        <lane id=":NE_Center_6_2" index="2" speed="13.89" length="27.20" shape="263.60,251.60 236.40,251.60"/>
    </edge>
    <edge id=":NE_Center_9" function="internal">
        <lane id=":NE_Center_9_0" index="0" speed="10.36" length="5.01" shape="263.60,251.60 258.64,250.89"/>
    </edge>
    <edge id=":NE_Center_21" function="internal">
        <lane id=":NE_Center_21_0" index="0" speed="10.36" length="19.50" shape="258.64,250.89 256.95,250.65 252.20,247.80 249.35,243.05 248.40,236.40"/>
    </edge>
    <edge id=":NE_Center_10" function="internal">
        <lane id=":NE_Center_10_0" index="0" speed="6.51" length="9.03" shape="258.00,236.40 258.35,238.85 259.40,240.60 261.15,241.65 263.60,242.00"/>
    </edge>
    <edge id=":NE_Center_11" function="internal">
        <lane id=":NE_Center_11_0" index="0" speed="13.89" length="27.20" shape="258.00,236.40 258.00,263.60"/>
        <lane id=":NE_Center_11_1" index="1" speed="13.89" length="27.20" shape="254.80,236.40 254.80,263.60"/>
        <lane id=":NE_Center_11_2" index="2" speed="13.89" length="27.20" shape="251.60,236.40 251.60,263.60"/>
    </edge>
    <edge id=":NE_Center_14" function="internal">
        <lane id=":NE_Center_14_0" index="0" speed="10.36" length="5.01" shape="251.60,236.40 250.89,241.36"/>
    </edge>
    <edge id=":NE_Center_22" function="internal">
        <lane id=":NE_Center_22_0" index="0" speed="10.36" length="19.50" shape="250.89,241.36 250.65,243.05 247.80,247.80 243.05,250.65 236.40,251.60"/>
    </edge>
    <edge id=":NE_Center_15" function="internal">
        <lane id=":NE_Center_15_0" index="0" speed="6.51" length="9.03" shape="236.40,242.00 238.85,241.65 240.60,240.60 241.65,238.85 242.00,236.40"/>
    </edge>
    <edge id=":NE_Center_16" function="internal">
        <lane id=":NE_Center_16_0" index="0" speed="13.89" length="27.20" shape="236.40,242.00 263.60,242.00"/>
        <lane id=":NE_Center_16_1" index="1" speed="13.89" length="27.20" shape="236.40,245.20 263.60,245.20"/>
        <lane id=":NE_Center_16_2" index="2" speed="13.89" length="27.20" shape="236.40,248.40 263.60,248.40"/>
    </edge>
    <edge id=":NE_Center_19" function="internal">
        <lane id=":NE_Center_19_0" index="0" speed="10.36" length="5.01" shape="236.40,248.40 241.36,249.11"/>
    </edge>
    <edge id=":NE_Center_23" function="internal">
        <lane id=":NE_Center_23_0" index="0" speed="10.36" length="19.50" shape="241.36,249.11 243.05,249.35 247.80,252.20 250.65,256.95 251.60,263.60"/>
    </edge>
    <edge id=":NW_Center_0" function="internal">
        <lane id=":NW_Center_0_0" index="0" speed="6.51" length="9.03" shape="-258.00,263.60 -258.35,261.15 -259.40,259.40 -261.15,258.35 -263.60,258.00"/>
    </edge>
    <edge id=":NW_Center_1" function="internal">
        <lane id=":NW_Center_1_0" index="0" speed="13.89" length="27.20" shape="-258.00,263.60 -258.00,236.40"/>
        <lane id=":NW_Center_1_1" index="1" speed="13.89" length="27.20" shape="-254.80,263.60 -254.80,236.40"/>
        <lane id=":NW_Center_1_2" index="2" speed="13.89" length="27.20" shape="-251.60,263.60 -251.60,236.40"/>
    </edge>
    <edge id=":NW_Center_4" function="internal">
        <lane id=":NW_Center_4_0" index="0" speed="10.36" length="5.01" shape="-251.60,263.60 -250.89,258.64"/>
    </edge>
    <edge id=":NW_Center_20" function="internal">
        <lane id=":NW_Center_20_0" index="0" speed="10.36" length="19.50" shape="-250.89,258.64 -250.65,256.95 -247.80,252.20 -243.05,249.35 -236.40,248.40"/>
    </edge>
    <edge id=":NW_Center_5" function="internal">
        <lane id=":NW_Center_5_0" index="0" speed="6.51" length="9.03" shape="-236.40,258.00 -238.85,258.35 -240.60,259.40 -241.65,261.15 -242.00,263.60"/>
    </edge>
    <edge id=":NW_Center_6" function="internal">
        <lane id=":NW_Center_6_0" index="0" speed="13.89" length="27.20" shape="-236.40,258.00 -263.60,258.00"/>
        <lane id=":NW_Center_6_1" index="1" speed="13.89" length="27.20" shape="-236.40,254.80 -263.60,254.80"/>
        <lane id=":NW_Center_6_2" index="2" speed="13.89" length="27.20" shape="-236.40,251.60 -263.60,251.60"/>
    </edge>
    <edge id=":NW_Center_9" function="internal">
        <lane id=":NW_Center_9_0" index="0" speed="10.36" length="5.01" shape="-236.40,251.60 -241.36,250.89"/>
    </edge>
    <edge id=":NW_Center_21" function="internal">
        <lane id=":NW_Center_21_0" index="0" speed="10.36" length="19.50" shape="-241.36,250.89 -243.05,250.65 -247.80,247.80 -250.65,243.05 -251.60,236.40"/>
    </edge>
    <edge id=":NW_Center_10" function="internal">
        <lane id=":NW_Center_10_0" index="0" speed="6.51" length="9.03" shape="-242.00,236.40 -241.65,238.85 -240.60,240.60 -238.85,241.65 -236.40,242.00"/>
    </edge>
    <edge id=":NW_Center_11" function="internal">
        <lane id=":NW_Center_11_0" index="0" speed="13.89" length="27.20" shape="-242.00,236.40 -242.00,263.60"/>
        <lane id=":NW_Center_11_1" index="1" speed="13.89" length="27.20" shape="-245.20,236.40 -245.20,263.60"/>
        <lane id=":NW_Center_11_2" index="2" speed="13.89" length="27.20" shape="-248.40,236.40 -248.40,263.60"/>
    </edge>
    <edge id=":NW_Center_14" function="internal">
        <lane id=":NW_Center_14_0" index="0" speed="10.36" length="5.01" shape="-248.40,236.40 -249.11,241.36"/>
    </edge>
    <edge id=":NW_Center_22" function="internal">
        <lane id=":NW_Center_22_0" index="0" speed="10.36" length="19.50" shape="-249.11,241.36 -249.35,243.05 -252.20,247.80 -256.95,250.65 -263.60,251.60"/>
    </edge>
    <edge id=":NW_Center_15" function="internal">
        <lane id=":NW_Center_15_0" index="0" speed="6.51" length="9.03" shape="-263.60,242.00 -261.15,241.65 -259.40,240.60 -258.35,238.85 -258.00,236.40"/>
    </edge>
    <edge id=":NW_Center_16" function="internal">
        <lane id=":NW_Center_16_0" index="0" speed="13.89" length="27.20" shape="-263.60,242.00 -236.40,242.00"/>
        <lane id=":NW_Center_16_1" index="1" speed="13.89" length="27.20" shape="-263.60,245.20 -236.40,245.20"/>
        <lane id=":NW_Center_16_2" index="2" speed="13.89" length="27.20" shape="-263.60,248.40 -236.40,248.40"/>
    </edge>
    <edge id=":NW_Center_19" function="internal">
        <lane id=":NW_Center_19_0" index="0" speed="10.36" length="5.01" shape="-263.60,248.40 -258.64,249.11"/>
    </edge>
    <edge id=":NW_Center_23" function="internal">
        <lane id=":NW_Center_23_0" index="0" speed="10.36" length="19.50" shape="-258.64,249.11 -256.95,249.35 -252.20,252.20 -249.35,256.95 -248.40,263.60"/>
    </edge>
    <edge id=":SE_Center_0" function="internal">
        <lane id=":SE_Center_0_0" index="0" speed="6.51" length="9.03" shape="242.00,-236.40 241.65,-238.85 240.60,-240.60 238.85,-241.65 236.40,-242.00"/>
    </edge>
    <edge id=":SE_Center_1" function="internal">
        <lane id=":SE_Center_1_0" index="0" speed="13.89" length="27.20" shape="242.00,-236.40 242.00,-263.60"/>
        <lane id=":SE_Center_1_1" index="1" speed="13.89" length="27.20" shape="245.20,-236.40 245.20,-263.60"/>
        <lane id=":SE_Center_1_2" index="2" speed="13.89" length="27.20" shape="248.40,-236.40 248.40,-263.60"/>
    </edge>
    <edge id=":SE_Center_4" function="internal">
        <lane id=":SE_Center_4_0" index="0" speed="10.36" length="5.01" shape="248.40,-236.40 249.11,-241.36"/>
    </edge>
    <edge id=":SE_Center_20" function="internal">
        <lane id=":SE_Center_20_0" index="0" speed="10.36" length="19.50" shape="249.11,-241.36 249.35,-243.05 252.20,-247.80 256.95,-250.65 263.60,-251.60"/>
    </edge>
    <edge id=":SE_Center_5" function="internal">
        <lane id=":SE_Center_5_0" index="0" speed="6.51" length="9.03" shape="263.60,-242.00 261.15,-241.65 259.40,-240.60 258.35,-238.85 258.00,-236.40"/>
    </edge>
    <edge id=":SE_Center_6" function="internal">
        <lane id=":SE_Center_6_0" index="0" speed="13.89" length="27.20" shape="263.60,-242.00 236.40,-242.00"/>
        <lane id=":SE_Center_6_1" index="1" speed="13.89" length="27.20" shape="263.60,-245.20 236.40,-245.20"/>
        <lane id=":SE_Center_6_2" index="2" speed="13.89" length="27.20" shape="263.60,-248.40 236.40,-248.40"/>
    </edge>
    <edge id=":SE_Center_9" function="internal">
        <lane id=":SE_Center_9_0" index="0" speed="10.36" length="5.01" shape="263.60,-248.40 258.64,-249.11"/>
    </edge>
    <edge id=":SE_Center_21" function="internal">
        <lane id=":SE_Center_21_0" index="0" speed="10.36" length="19.50" shape="258.64,-249.11 256.95,-249.35 252.20,-252.20 249.35,-256.95 248.40,-263.60"/>
    </edge>
    <edge id=":SE_Center_10" function="internal">
        <lane id=":SE_Center_10_0" index="0" speed="6.51" length="9.03" shape="258.00,-263.60 258.35,-261.15 259.40,-259.40 261.15,-258.35 263.60,-258.00"/>
    </edge>
    <edge id=":SE_Center_11" function="internal">
        <lane id=":SE_Center_11_0" index="0" speed="13.89" length="27.20" shape="258.00,-263.60 258.00,-236.40"/>
        <lane id=":SE_Center_11_1" index="1" speed="13.89" length="27.20" shape="254.80,-263.60 254.80,-236.40"/>
        <lane id=":SE_Center_11_2" index="2" speed="13.89" length="27.20" shape="251.60,-263.60 251.60,-236.40"/>
    </edge>
    <edge id=":SE_Center_14" function="internal">
        <lane id=":SE_Center_14_0" index="0" speed="10.36" length="5.01" shape="251.60,-263.60 250.89,-258.64"/>
    </edge>
    <edge id=":SE_Center_22" function="internal">
        <lane id=":SE_Center_22_0" index="0" speed="10.36" length="19.50" shape="250.89,-258.64 250.65,-256.95 247.80,-252.20 243.05,-249.35 236.40,-248.40"/>
    </edge>
    <edge id=":SE_Center_15" function="internal">
        <lane id=":SE_Center_15_0" index="0" speed="6.51" length="9.03" shape="236.40,-258.00 238.85,-258.35 240.60,-259.40 241.65,-261.15 242.00,-263.60"/>
    </edge>
    <edge id=":SE_Center_16" function="internal">
        <lane id=":SE_Center_16_0" index="0" speed="13.89" length="27.20" shape="236.40,-258.00 263.60,-258.00"/>
        <lane id=":SE_Center_16_1" index="1" speed="13.89" length="27.20" shape="236.40,-254.80 263.60,-254.80"/>
        <lane id=":SE_Center_16_2" index="2" speed="13.89" length="27.20" shape="236.40,-251.60 263.60,-251.60"/>
    </edge>
    <edge id=":SE_Center_19" function="internal">
        <lane id=":SE_Center_19_0" index="0" speed="10.36" length="5.01" shape="236.40,-251.60 241.36,-250.89"/>
    </edge>
    <edge id=":SE_Center_23" function="internal">
        <lane id=":SE_Center_23_0" index="0" speed="10.36" length="19.50" shape="241.36,-250.89 243.05,-250.65 247.80,-247.80 250.65,-243.05 251.60,-236.40"/>
    </edge>
    <edge id=":SW_Center_0" function="internal">
        <lane id=":SW_Center_0_0" index="0" speed="6.51" length="9.03" shape="-258.00,-236.40 -258.35,-238.85 -259.40,-240.60 -261.15,-241.65 -263.60,-242.00"/>
    </edge>
    <edge id=":SW_Center_1" function="internal">
        <lane id=":SW_Center_1_0" index="0" speed="13.89" length="27.20" shape="-258.00,-236.40 -258.00,-263.60"/>
        <lane id=":SW_Center_1_1" index="1" speed="13.89" length="27.20" shape="-254.80,-236.40 -254.80,-263.60"/>
        <lane id=":SW_Center_1_2" index="2" speed="13.89" length="27.20" shape="-251.60,-236.40 -251.60,-263.60"/>
    </edge>
    <edge id=":SW_Center_4" function="internal">
        <lane id=":SW_Center_4_0" index="0" speed="10.36" length="5.01" shape="-251.60,-236.40 -250.89,-241.36"/>
    </edge>
    <edge id=":SW_Center_20" function="internal">
        <lane id=":SW_Center_20_0" index="0" speed="10.36" length="19.50" shape="-250.89,-241.36 -250.65,-243.05 -247.80,-247.80 -243.05,-250.65 -236.40,-251.60"/>
    </edge>
    <edge id=":SW_Center_5" function="internal">
        <lane id=":SW_Center_5_0" index="0" speed="6.51" length="9.03" shape="-236.40,-242.00 -238.85,-241.65 -240.60,-240.60 -241.65,-238.85 -242.00,-236.40"/>
    </edge>
    <edge id=":SW_Center_6" function="internal">
        <lane id=":SW_Center_6_0" index="0" speed="13.89" length="27.20" shape="-236.40,-242.00 -263.60,-242.00"/>
        <lane id=":SW_Center_6_1" index="1" speed="13.89" length="27.20" shape="-236.40,-245.20 -263.60,-245.20"/>
        <lane id=":SW_Center_6_2" index="2" speed="13.89" length="27.20" shape="-236.40,-248.40 -263.60,-248.40"/>
    </edge>
    <edge id=":SW_Center_9" function="internal">
        <lane id=":SW_Center_9_0" index="0" speed="10.36" length="5.01" shape="-236.40,-248.40 -241.36,-249.11"/>
    </edge>
    <edge id=":SW_Center_21" function="internal">
        <lane id=":SW_Center_21_0" index="0" speed="10.36" length="19.50" shape="-241.36,-249.11 -243.05,-249.35 -247.80,-252.20 -250.65,-256.95 -251.60,-263.60"/>
    </edge>
    <edge id=":SW_Center_10" function="internal">
        <lane id=":SW_Center_10_0" index="0" speed="6.51" length="9.03" shape="-242.00,-263.60 -241.65,-261.15 -240.60,-259.40 -238.85,-258.35 -236.40,-258.00"/>
    </edge>
    <edge id=":SW_Center_11" function="internal">
        <lane id=":SW_Center_11_0" index="0" speed="13.89" length="27.20" shape="-242.00,-263.60 -242.00,-236.40"/>
        <lane id=":SW_Center_11_1" index="1" speed="13.89" length="27.20" shape="-245.20,-263.60 -245.20,-236.40"/>
        <lane id=":SW_Center_11_2" index="2" speed="13.89" length="27.20" shape="-248.40,-263.60 -248.40,-236.40"/>
    </edge>
    <edge id=":SW_Center_14" function="internal">
        <lane id=":SW_Center_14_0" index="0" speed="10.36" length="5.01" shape="-248.40,-263.60 -249.11,-258.64"/>
    </edge>
    <edge id=":SW_Center_22" function="internal">
        <lane id=":SW_Center_22_0" index="0" speed="10.36" length="19.50" shape="-249.11,-258.64 -249.35,-256.95 -252.20,-252.20 -256.95,-249.35 -263.60,-248.40"/>
    </edge>
    <edge id=":SW_Center_15" function="internal">
        <lane id=":SW_Center_15_0" index="0" speed="6.51" length="9.03" shape="-263.60,-258.00 -261.15,-258.35 -259.40,-259.40 -258.35,-261.15 -258.00,-263.60"/>
    </edge>
    <edge id=":SW_Center_16" function="internal">
        <lane id=":SW_Center_16_0" index="0" speed="13.89" length="27.20" shape="-263.60,-258.00 -236.40,-258.00"/>
        <lane id=":SW_Center_16_1" index="1" speed="13.89" length="27.20" shape="-263.60,-254.80 -236.40,-254.80"/>
        <lane id=":SW_Center_16_2" index="2" speed="13.89" length="27.20" shape="-263.60,-251.60 -236.40,-251.60"/>
    </edge>
    <edge id=":SW_Center_19" function="internal">
        <lane id=":SW_Center_19_0" index="0" speed="10.36" length="5.01" shape="-263.60,-251.60 -258.64,-250.89"/>
    </edge>
    <edge id=":SW_Center_23" function="internal">
        <lane id=":SW_Center_23_0" index="0" speed="10.36" length="19.50" shape="-258.64,-250.89 -256.95,-250.65 -252.20,-247.80 -249.35,-243.05 -248.40,-236.40"/>
    </edge>

    <edge id="-E0" from="NW_Center" to="WN" priority="-1">
        <lane id="-E0_0" index="0" speed="13.89" length="236.40" shape="-263.60,258.00 -500.00,258.00"/>
        <lane id="-E0_1" index="1" speed="13.89" length="236.40" shape="-263.60,254.80 -500.00,254.80"/>
        <lane id="-E0_2" index="2" speed="13.89" length="236.40" shape="-263.60,251.60 -500.00,251.60"/>
    </edge>
    <edge id="-E1" from="NE_Center" to="NW_Center" priority="-1">
        <lane id="-E1_0" index="0" speed="13.89" length="472.80" shape="236.40,258.00 -236.40,258.00"/>
        <lane id="-E1_1" index="1" speed="13.89" length="472.80" shape="236.40,254.80 -236.40,254.80"/>
        <lane id="-E1_2" index="2" speed="13.89" length="472.80" shape="236.40,251.60 -236.40,251.60"/>
    </edge>
    <edge id="-E10" from="SW" to="SW_Center" priority="-1">
        <lane id="-E10_0" index="0" speed="13.89" length="236.40" shape="-242.00,-500.00 -242.00,-263.60"/>
        <lane id="-E10_1" index="1" speed="13.89" length="236.40" shape="-245.20,-500.00 -245.20,-263.60"/>
        <lane id="-E10_2" index="2" speed="13.89" length="236.40" shape="-248.40,-500.00 -248.40,-263.60"/>
    </edge>
    <edge id="-E11" from="WS" to="SW_Center" priority="-1">
        <lane id="-E11_0" index="0" speed="13.89" length="236.40" shape="-500.00,-258.00 -263.60,-258.00"/>
        <lane id="-E11_1" index="1" speed="13.89" length="236.40" shape="-500.00,-254.80 -263.60,-254.80"/>
        <lane id="-E11_2" index="2" speed="13.89" length="236.40" shape="-500.00,-251.60 -263.60,-251.60"/>
    </edge>
    <edge id="-E2" from="NW_Center" to="NW" priority="-1">
        <lane id="-E2_0" index="0" speed="13.89" length="236.40" shape="-242.00,263.60 -242.00,500.00"/>
        <lane id="-E2_1" index="1" speed="13.89" length="236.40" shape="-245.20,263.60 -245.20,500.00"/>
        <lane id="-E2_2" index="2" speed="13.89" length="236.40" shape="-248.40,263.60 -248.40,500.00"/>
    </edge>
    <edge id="-E3" from="SW_Center" to="NW_Center" priority="-1">
        <lane id="-E3_0" index="0" speed="13.89" length="472.80" shape="-242.00,-236.40 -242.00,236.40"/>
        <lane id="-E3_1" index="1" speed="13.89" length="472.80" shape="-245.20,-236.40 -245.20,236.40"/>
        <lane id="-E3_2" index="2" speed="13.89" length="472.80" shape="-248.40,-236.40 -248.40,236.40"/>
    </edge>
    <edge id="-E4" from="EN" to="NE_Center" priority="-1">
        <lane id="-E4_0" index="0" speed="13.89" length="236.40" shape="500.00,258.00 263.60,258.00"/>
        <lane id="-E4_1" index="1" speed="13.89" length="236.40" shape="500.00,254.80 263.60,254.80"/>
        <lane id="-E4_2" index="2" speed="13.89" length="236.40" shape="500.00,251.60 263.60,251.60"/>
    </edge>
    <edge id="-E5" from="NE" to="NE_Center" priority="-1">
        <lane id="-E5_0" index="0" speed="13.89" length="236.40" shape="242.00,500.00 242.00,263.60"/>
        <lane id="-E5_1" index="1" speed="13.89" length="236.40" shape="245.20,500.00 245.20,263.60"/>
        <lane id="-E5_2" index="2" speed="13.89" length="236.40" shape="248.40,500.00 248.40,263.60"/>
    </edge>
    <edge id="-E6" from="SE_Center" to="NE_Center" priority="-1">
        <lane id="-E6_0" index="0" speed="13.89" length="472.80" shape="258.00,-236.40 258.00,236.40"/>
        <lane id="-E6_1" index="1" speed="13.89" length="472.80" shape="254.80,-236.40 254.80,236.40"/>
        <lane id="-E6_2" index="2" speed="13.89" length="472.80" shape="251.60,-236.40 251.60,236.40"/>
    </edge>
    <edge id="-E7" from="SW_Center" to="SE_Center" priority="-1">
        <lane id="-E7_0" index="0" speed="13.89" length="472.80" shape="-236.40,-258.00 236.40,-258.00"/>
        <lane id="-E7_1" index="1" speed="13.89" length="472.80" shape="-236.40,-254.80 236.40,-254.80"/>
        <lane id="-E7_2" index="2" speed="13.89" length="472.80" shape="-236.40,-251.60 236.40,-251.60"/>
    </edge>
    <edge id="-E8" from="EW" to="SE_Center" priority="-1">
        <lane id="-E8_0" index="0" speed="13.89" length="236.40" shape="500.00,-242.00 263.60,-242.00"/>
        <lane id="-E8_1" index="1" speed="13.89" length="236.40" shape="500.00,-245.20 263.60,-245.20"/>
        <lane id="-E8_2" index="2" speed="13.89" length="236.40" shape="500.00,-248.40 263.60,-248.40"/>
    </edge>
    <edge id="-E9" from="SE" to="SE_Center" priority="-1">
        <lane id="-E9_0" index="0" speed="13.89" length="236.40" shape="258.00,-500.00 258.00,-263.60"/>
        <lane id="-E9_1" index="1" speed="13.89" length="236.40" shape="254.80,-500.00 254.80,-263.60"/>
        <lane id="-E9_2" index="2" speed="13.89" length="236.40" shape="251.60,-500.00 251.60,-263.60"/>
    </edge>
    <edge id="E0" from="WN" to="NW_Center" priority="-1">
        <lane id="E0_0" index="0" speed="13.89" length="236.40" shape="-500.00,242.00 -263.60,242.00"/>
        <lane id="E0_1" index="1" speed="13.89" length="236.40" shape="-500.00,245.20 -263.60,245.20"/>
        <lane id="E0_2" index="2" speed="13.89" length="236.40" shape="-500.00,248.40 -263.60,248.40"/>
    </edge>
    <edge id="E1" from="NW_Center" to="NE_Center" priority="-1">
        <lane id="E1_0" index="0" speed="13.89" length="472.80" shape="-236.40,242.00 236.40,242.00"/>
        <lane id="E1_1" index="1" speed="13.89" length="472.80" shape="-236.40,245.20 236.40,245.20"/>
        <lane id="E1_2" index="2" speed="13.89" length="472.80" shape="-236.40,248.40 236.40,248.40"/>
    </edge>
    <edge id="E10" from="SW_Center" to="SW" priority="-1">
        <lane id="E10_0" index="0" speed="13.89" length="236.40" shape="-258.00,-263.60 -258.00,-500.00"/>
        <lane id="E10_1" index="1" speed="13.89" length="236.40" shape="-254.80,-263.60 -254.80,-500.00"/>
        <lane id="E10_2" index="2" speed="13.89" length="236.40" shape="-251.60,-263.60 -251.60,-500.00"/>
    </edge>
    <edge id="E11" from="SW_Center" to="WS" priority="-1">
        <lane id="E11_0" index="0" speed="13.89" length="236.40" shape="-263.60,-242.00 -500.00,-242.00"/>
        <lane id="E11_1" index="1" speed="13.89" length="236.40" shape="-263.60,-245.20 -500.00,-245.20"/>
        <lane id="E11_2" index="2" speed="13.89" length="236.40" shape="-263.60,-248.40 -500.00,-248.40"/>
    </edge>
    <edge id="E2" from="NW" to="NW_Center" priority="-1">
        <lane id="E2_0" index="0" speed="13.89" length="236.40" shape="-258.00,500.00 -258.00,263.60"/>
        <lane id="E2_1" index="1" speed="13.89" length="236.40" shape="-254.80,500.00 -254.80,263.60"/>
        <lane id="E2_2" index="2" speed="13.89" length="236.40" shape="-251.60,500.00 -251.60,263.60"/>
    </edge>
    <edge id="E3" from="NW_Center" to="SW_Center" priority="-1">
        <lane id="E3_0" index="0" speed="13.89" length="472.80" shape="-258.00,236.40 -258.00,-236.40"/>
        <lane id="E3_1" index="1" speed="13.89" length="472.80" shape="-254.80,236.40 -254.80,-236.40"/>
        <lane id="E3_2" index="2" speed="13.89" length="472.80" shape="-251.60,236.40 -251.60,-236.40"/>
    </edge>
    <edge id="E4" from="NE_Center" to="EN" priority="-1">
        <lane id="E4_0" index="0" speed="13.89" length="236.40" shape="263.60,242.00 500.00,242.00"/>
        <lane id="E4_1" index="1" speed="13.89" length="236.40" shape="263.60,245.20 500.00,245.20"/>
        <lane id="E4_2" index="2" speed="13.89" length="236.40" shape="263.60,248.40 500.00,248.40"/>
    </edge>
    <edge id="E5" from="NE_Center" to="NE" priority="-1">
        <lane id="E5_0" index="0" speed="13.89" length="236.40" shape="258.00,263.60 258.00,500.00"/>
        <lane id="E5_1" index="1" speed="13.89" length="236.40" shape="254.80,263.60 254.80,500.00"/>
        <lane id="E5_2" index="2" speed="13.89" length="236.40" shape="251.60,263.60 251.60,500.00"/>
    </edge>
    <edge id="E6" from="NE_Center" to="SE_Center" priority="-1">
        <lane id="E6_0" index="0" speed="13.89" length="472.80" shape="242.00,236.40 242.00,-236.40"/>
        <lane id="E6_1" index="1" speed="13.89" length="472.80" shape="245.20,236.40 245.20,-236.40"/>
        <lane id="E6_2" index="2" speed="13.89" length="472.80" shape="248.40,236.40 248.40,-236.40"/>
    </edge>
    <edge id="E7" from="SE_Center" to="SW_Center" priority="-1">
        <lane id="E7_0" index="0" speed="13.89" length="472.80" shape="236.40,-242.00 -236.40,-242.00"/>
        <lane id="E7_1" index="1" speed="13.89" length="472.80" shape="236.40,-245.20 -236.40,-245.20"/>
        <lane id="E7_2" index="2" speed="13.89" length="472.80" shape="236.40,-248.40 -236.40,-248.40"/>
    </edge>
    <edge id="E8" from="SE_Center" to="EW" priority="-1">
        <lane id="E8_0" index="0" speed="13.89" length="236.40" shape="263.60,-258.00 500.00,-258.00"/>
        <lane id="E8_1" index="1" speed="13.89" length="236.40" shape="263.60,-254.80 500.00,-254.80"/>
        <lane id="E8_2" index="2" speed="13.89" length="236.40" shape="263.60,-251.60 500.00,-251.60"/>
    </edge>
    <edge id="E9" from="SE_Center" to="SE" priority="-1">
        <lane id="E9_0" index="0" speed="13.89" length="236.40" shape="242.00,-263.60 242.00,-500.00"/>
        <lane id="E9_1" index="1" speed="13.89" length="236.40" shape="245.20,-263.60 245.20,-500.00"/>
        <lane id="E9_2" index="2" speed="13.89" length="236.40" shape="248.40,-263.60 248.40,-500.00"/>
    </edge>

    <tlLogic id="NE_Center" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGGgrrrrrGGGGgrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGGgrrrrrGGGGg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>
    <tlLogic id="NW_Center" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGGgrrrrrGGGGgrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGGgrrrrrGGGGg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>
    <tlLogic id="SE_Center" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGGgrrrrrGGGGgrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGGgrrrrrGGGGg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>
    <tlLogic id="SW_Center" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGGgrrrrrGGGGgrrrrr"/>
        <phase duration="3"  state="yyyyyrrrrryyyyyrrrrr"/>
        <phase duration="42" state="rrrrrGGGGgrrrrrGGGGg"/>
        <phase duration="3"  state="rrrrryyyyyrrrrryyyyy"/>
    </tlLogic>

    <junction id="EN" type="dead_end" x="500.00" y="250.00" incLanes="E4_0 E4_1 E4_2" intLanes="" shape="500.00,250.00 500.00,240.40 500.00,250.00"/>
    <junction id="EW" type="dead_end" x="500.00" y="-250.00" incLanes="E8_0 E8_1 E8_2" intLanes="" shape="500.00,-250.00 500.00,-259.60 500.00,-250.00"/>
    <junction id="NE" type="dead_end" x="250.00" y="500.00" incLanes="E5_0 E5_1 E5_2" intLanes="" shape="250.00,500.00 259.60,500.00 250.00,500.00"/>
    <junction id="NE_Center" type="traffic_light" x="250.00" y="250.00" incLanes="-E5_0 -E5_1 -E5_2 -E4_0 -E4_1 -E4_2 -E6_0 -E6_1 -E6_2 E1_0 E1_1 E1_2" intLanes=":NE_Center_0_0 :NE_Center_1_0 :NE_Center_1_1 :NE_Center_1_2 :NE_Center_20_0 :NE_Center_5_0 :NE_Center_6_0 :NE_Center_6_1 :NE_Center_6_2 :NE_Center_21_0 :NE_Center_10_0 :NE_Center_11_0 :NE_Center_11_1 :NE_Center_11_2 :NE_Center_22_0 :NE_Center_15_0 :NE_Center_16_0 :NE_Center_16_1 :NE_Center_16_2 :NE_Center_23_0" shape="240.40,263.60 259.60,263.60 260.04,261.38 260.60,260.60 261.38,260.04 262.38,259.71 263.60,259.60 263.60,240.40 261.38,239.96 260.60,239.40 260.04,238.62 259.71,237.62 259.60,236.40 240.40,236.40 239.96,238.62 239.40,239.40 238.62,239.96 237.62,240.29 236.40,240.40 236.40,259.60 238.62,260.04 239.40,260.60 239.96,261.38 240.29,262.38">
        <request index="0"  response="00000000000000000000" foes="00000000000111000000" cont="0"/>
        <request index="1"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="2"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="3"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="4"  response="10000011101000000000" foes="11110011101111000000" cont="1"/>
        <request index="5"  response="00000011100000000000" foes="00000011100000000000" cont="0"/>
        <request index="6"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="7"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="8"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="9"  response="01110111100000011110" foes="01110111100000011110" cont="1"/>
        <request index="10" response="00000000000000000000" foes="01110000000000000000" cont="0"/>
        <request index="11" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="12" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="13" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="14" response="10000000001000001110" foes="11110000001111001110" cont="1"/>
        <request index="15" response="00000000000000001110" foes="00000000000000001110" cont="0"/>
        <request index="16" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="17" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="18" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="19" response="00000111100111011110" foes="00000111100111011110" cont="1"/>
    </junction>
    <junction id="NW" type="dead_end" x="-250.00" y="500.00" incLanes="-E2_0 -E2_1 -E2_2" intLanes="" shape="-250.00,500.00 -240.40,500.00 -250.00,500.00"/>
    <junction id="NW_Center" type="traffic_light" x="-250.00" y="250.00" incLanes="E2_0 E2_1 E2_2 -E1_0 -E1_1 -E1_2 -E3_0 -E3_1 -E3_2 E0_0 E0_1 E0_2" intLanes=":NW_Center_0_0 :NW_Center_1_0 :NW_Center_1_1 :NW_Center_1_2 :NW_Center_20_0 :NW_Center_5_0 :NW_Center_6_0 :NW_Center_6_1 :NW_Center_6_2 :NW_Center_21_0 :NW_Center_10_0 :NW_Center_11_0 :NW_Center_11_1 :NW_Center_11_2 :NW_Center_22_0 :NW_Center_15_0 :NW_Center_16_0 :NW_Center_16_1 :NW_Center_16_2 :NW_Center_23_0" shape="-259.60,263.60 -240.40,263.60 -239.96,261.38 -239.40,260.60 -238.62,260.04 -237.62,259.71 -236.40,259.60 -236.40,240.40 -238.62,239.96 -239.40,239.40 -239.96,238.62 -240.29,237.62 -240.40,236.40 -259.60,236.40 -260.04,238.62 -260.60,239.40 -261.38,239.96 -262.38,240.29 -263.60,240.40 -263.60,259.60 -261.38,260.04 -260.60,260.60 -260.04,261.38 -259.71,262.38">
        <request index="0"  response="00000000000000000000" foes="00000000000111000000" cont="0"/>
        <request index="1"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="2"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="3"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="4"  response="10000011101000000000" foes="11110011101111000000" cont="1"/>
        <request index="5"  response="00000011100000000000" foes="00000011100000000000" cont="0"/>
        <request index="6"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="7"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="8"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="9"  response="01110111100000011110" foes="01110111100000011110" cont="1"/>
        <request index="10" response="00000000000000000000" foes="01110000000000000000" cont="0"/>
        <request index="11" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="12" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="13" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="14" response="10000000001000001110" foes="11110000001111001110" cont="1"/>
        <request index="15" response="00000000000000001110" foes="00000000000000001110" cont="0"/>
        <request index="16" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="17" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="18" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="19" response="00000111100111011110" foes="00000111100111011110" cont="1"/>
    </junction>
    <junction id="SE" type="dead_end" x="250.00" y="-500.00" incLanes="E9_0 E9_1 E9_2" intLanes="" shape="250.00,-500.00 240.40,-500.00 250.00,-500.00"/>
    <junction id="SE_Center" type="traffic_light" x="250.00" y="-250.00" incLanes="E6_0 E6_1 E6_2 -E8_0 -E8_1 -E8_2 -E9_0 -E9_1 -E9_2 -E7_0 -E7_1 -E7_2" intLanes=":SE_Center_0_0 :SE_Center_1_0 :SE_Center_1_1 :SE_Center_1_2 :SE_Center_20_0 :SE_Center_5_0 :SE_Center_6_0 :SE_Center_6_1 :SE_Center_6_2 :SE_Center_21_0 :SE_Center_10_0 :SE_Center_11_0 :SE_Center_11_1 :SE_Center_11_2 :SE_Center_22_0 :SE_Center_15_0 :SE_Center_16_0 :SE_Center_16_1 :SE_Center_16_2 :SE_Center_23_0" shape="240.40,-236.40 259.60,-236.40 260.04,-238.62 260.60,-239.40 261.38,-239.96 262.38,-240.29 263.60,-240.40 263.60,-259.60 261.38,-260.04 260.60,-260.60 260.04,-261.38 259.71,-262.38 259.60,-263.60 240.40,-263.60 239.96,-261.38 239.40,-260.60 238.62,-260.04 237.62,-259.71 236.40,-259.60 236.40,-240.40 238.62,-239.96 239.40,-239.40 239.96,-238.62 240.29,-237.62">
        <request index="0"  response="00000000000000000000" foes="00000000000111000000" cont="0"/>
        <request index="1"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="2"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="3"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="4"  response="10000011101000000000" foes="11110011101111000000" cont="1"/>
        <request index="5"  response="00000011100000000000" foes="00000011100000000000" cont="0"/>
        <request index="6"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="7"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="8"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="9"  response="01110111100000011110" foes="01110111100000011110" cont="1"/>
        <request index="10" response="00000000000000000000" foes="01110000000000000000" cont="0"/>
        <request index="11" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="12" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="13" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="14" response="10000000001000001110" foes="11110000001111001110" cont="1"/>
        <request index="15" response="00000000000000001110" foes="00000000000000001110" cont="0"/>
        <request index="16" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="17" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="18" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="19" response="00000111100111011110" foes="00000111100111011110" cont="1"/>
    </junction>
    <junction id="SW" type="dead_end" x="-250.00" y="-500.00" incLanes="E10_0 E10_1 E10_2" intLanes="" shape="-250.00,-500.00 -259.60,-500.00 -250.00,-500.00"/>
    <junction id="SW_Center" type="traffic_light" x="-250.00" y="-250.00" incLanes="E3_0 E3_1 E3_2 E7_0 E7_1 E7_2 -E10_0 -E10_1 -E10_2 -E11_0 -E11_1 -E11_2" intLanes=":SW_Center_0_0 :SW_Center_1_0 :SW_Center_1_1 :SW_Center_1_2 :SW_Center_20_0 :SW_Center_5_0 :SW_Center_6_0 :SW_Center_6_1 :SW_Center_6_2 :SW_Center_21_0 :SW_Center_10_0 :SW_Center_11_0 :SW_Center_11_1 :SW_Center_11_2 :SW_Center_22_0 :SW_Center_15_0 :SW_Center_16_0 :SW_Center_16_1 :SW_Center_16_2 :SW_Center_23_0" shape="-259.60,-236.40 -240.40,-236.40 -239.96,-238.62 -239.40,-239.40 -238.62,-239.96 -237.62,-240.29 -236.40,-240.40 -236.40,-259.60 -238.62,-260.04 -239.40,-260.60 -239.96,-261.38 -240.29,-262.38 -240.40,-263.60 -259.60,-263.60 -260.04,-261.38 -260.60,-260.60 -261.38,-260.04 -262.38,-259.71 -263.60,-259.60 -263.60,-240.40 -261.38,-239.96 -260.60,-239.40 -260.04,-238.62 -259.71,-237.62">
        <request index="0"  response="00000000000000000000" foes="00000000000111000000" cont="0"/>
        <request index="1"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="2"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="3"  response="10000000001000000000" foes="11111100001111000000" cont="0"/>
        <request index="4"  response="10000011101000000000" foes="11110011101111000000" cont="1"/>
        <request index="5"  response="00000011100000000000" foes="00000011100000000000" cont="0"/>
        <request index="6"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="7"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="8"  response="00000111100000011111" foes="10000111100000011111" cont="0"/>
        <request index="9"  response="01110111100000011110" foes="01110111100000011110" cont="1"/>
        <request index="10" response="00000000000000000000" foes="01110000000000000000" cont="0"/>
        <request index="11" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="12" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="13" response="10000000001000000000" foes="11110000001111110000" cont="0"/>
        <request index="14" response="10000000001000001110" foes="11110000001111001110" cont="1"/>
        <request index="15" response="00000000000000001110" foes="00000000000000001110" cont="0"/>
        <request index="16" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="17" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="18" response="00000111110000011110" foes="00000111111000011110" cont="0"/>
        <request index="19" response="00000111100111011110" foes="00000111100111011110" cont="1"/>
    </junction>
    <junction id="WN" type="dead_end" x="-500.00" y="250.00" incLanes="-E0_0 -E0_1 -E0_2" intLanes="" shape="-500.00,250.00 -500.00,259.60 -500.00,250.00"/>
    <junction id="WS" type="dead_end" x="-500.00" y="-250.00" incLanes="E11_0 E11_1 E11_2" intLanes="" shape="-500.00,-250.00 -500.00,-240.40 -500.00,-250.00"/>

    <junction id=":NE_Center_20_0" type="internal" x="249.11" y="258.64" incLanes=":NE_Center_4_0 -E6_0 -E6_1 -E6_2" intLanes=":NE_Center_6_0 :NE_Center_6_1 :NE_Center_6_2 :NE_Center_9_0 :NE_Center_10_0 :NE_Center_11_0 :NE_Center_11_1 :NE_Center_11_2 :NE_Center_16_0 :NE_Center_16_1 :NE_Center_16_2 :NE_Center_19_0"/>
    <junction id=":NE_Center_21_0" type="internal" x="258.64" y="250.89" incLanes=":NE_Center_9_0 E1_0 E1_1 E1_2" intLanes=":NE_Center_1_0 :NE_Center_1_1 :NE_Center_1_2 :NE_Center_4_0 :NE_Center_11_0 :NE_Center_11_1 :NE_Center_11_2 :NE_Center_14_0 :NE_Center_15_0 :NE_Center_16_0 :NE_Center_16_1 :NE_Center_16_2"/>
    <junction id=":NE_Center_22_0" type="internal" x="250.89" y="241.36" incLanes=":NE_Center_14_0 -E5_0 -E5_1 -E5_2" intLanes=":NE_Center_0_0 :NE_Center_1_0 :NE_Center_1_1 :NE_Center_1_2 :NE_Center_6_0 :NE_Center_6_1 :NE_Center_6_2 :NE_Center_9_0 :NE_Center_16_0 :NE_Center_16_1 :NE_Center_16_2 :NE_Center_19_0"/>
    <junction id=":NE_Center_23_0" type="internal" x="241.36" y="249.11" incLanes=":NE_Center_19_0 -E4_0 -E4_1 -E4_2" intLanes=":NE_Center_1_0 :NE_Center_1_1 :NE_Center_1_2 :NE_Center_4_0 :NE_Center_5_0 :NE_Center_6_0 :NE_Center_6_1 :NE_Center_6_2 :NE_Center_11_0 :NE_Center_11_1 :NE_Center_11_2 :NE_Center_14_0"/>
    <junction id=":NW_Center_20_0" type="internal" x="-250.89" y="258.64" incLanes=":NW_Center_4_0 -E3_0 -E3_1 -E3_2" intLanes=":NW_Center_6_0 :NW_Center_6_1 :NW_Center_6_2 :NW_Center_9_0 :NW_Center_10_0 :NW_Center_11_0 :NW_Center_11_1 :NW_Center_11_2 :NW_Center_16_0 :NW_Center_16_1 :NW_Center_16_2 :NW_Center_19_0"/>
    <junction id=":NW_Center_21_0" type="internal" x="-241.36" y="250.89" incLanes=":NW_Center_9_0 E0_0 E0_1 E0_2" intLanes=":NW_Center_1_0 :NW_Center_1_1 :NW_Center_1_2 :NW_Center_4_0 :NW_Center_11_0 :NW_Center_11_1 :NW_Center_11_2 :NW_Center_14_0 :NW_Center_15_0 :NW_Center_16_0 :NW_Center_16_1 :NW_Center_16_2"/>
    <junction id=":NW_Center_22_0" type="internal" x="-249.11" y="241.36" incLanes=":NW_Center_14_0 E2_0 E2_1 E2_2" intLanes=":NW_Center_0_0 :NW_Center_1_0 :NW_Center_1_1 :NW_Center_1_2 :NW_Center_6_0 :NW_Center_6_1 :NW_Center_6_2 :NW_Center_9_0 :NW_Center_16_0 :NW_Center_16_1 :NW_Center_16_2 :NW_Center_19_0"/>
    <junction id=":NW_Center_23_0" type="internal" x="-258.64" y="249.11" incLanes=":NW_Center_19_0 -E1_0 -E1_1 -E1_2" intLanes=":NW_Center_1_0 :NW_Center_1_1 :NW_Center_1_2 :NW_Center_4_0 :NW_Center_5_0 :NW_Center_6_0 :NW_Center_6_1 :NW_Center_6_2 :NW_Center_11_0 :NW_Center_11_1 :NW_Center_11_2 :NW_Center_14_0"/>
    <junction id=":SE_Center_20_0" type="internal" x="249.11" y="-241.36" incLanes=":SE_Center_4_0 -E9_0 -E9_1 -E9_2" intLanes=":SE_Center_6_0 :SE_Center_6_1 :SE_Center_6_2 :SE_Center_9_0 :SE_Center_10_0 :SE_Center_11_0 :SE_Center_11_1 :SE_Center_11_2 :SE_Center_16_0 :SE_Center_16_1 :SE_Center_16_2 :SE_Center_19_0"/>
    <junction id=":SE_Center_21_0" type="internal" x="258.64" y="-249.11" incLanes=":SE_Center_9_0 -E7_0 -E7_1 -E7_2" intLanes=":SE_Center_1_0 :SE_Center_1_1 :SE_Center_1_2 :SE_Center_4_0 :SE_Center_11_0 :SE_Center_11_1 :SE_Center_11_2 :SE_Center_14_0 :SE_Center_15_0 :SE_Center_16_0 :SE_Center_16_1 :SE_Center_16_2"/>
    <junction id=":SE_Center_22_0" type="internal" x="250.89" y="-258.64" incLanes=":SE_Center_14_0 E6_0 E6_1 E6_2" intLanes=":SE_Center_0_0 :SE_Center_1_0 :SE_Center_1_1 :SE_Center_1_2 :SE_Center_6_0 :SE_Center_6_1 :SE_Center_6_2 :SE_Center_9_0 :SE_Center_16_0 :SE_Center_16_1 :SE_Center_16_2 :SE_Center_19_0"/>
    <junction id=":SE_Center_23_0" type="internal" x="241.36" y="-250.89" incLanes=":SE_Center_19_0 -E8_0 -E8_1 -E8_2" intLanes=":SE_Center_1_0 :SE_Center_1_1 :SE_Center_1_2 :SE_Center_4_0 :SE_Center_5_0 :SE_Center_6_0 :SE_Center_6_1 :SE_Center_6_2 :SE_Center_11_0 :SE_Center_11_1 :SE_Center_11_2 :SE_Center_14_0"/>
    <junction id=":SW_Center_20_0" type="internal" x="-250.89" y="-241.36" incLanes=":SW_Center_4_0 -E10_0 -E10_1 -E10_2" intLanes=":SW_Center_6_0 :SW_Center_6_1 :SW_Center_6_2 :SW_Center_9_0 :SW_Center_10_0 :SW_Center_11_0 :SW_Center_11_1 :SW_Center_11_2 :SW_Center_16_0 :SW_Center_16_1 :SW_Center_16_2 :SW_Center_19_0"/>
    <junction id=":SW_Center_21_0" type="internal" x="-241.36" y="-249.11" incLanes=":SW_Center_9_0 -E11_0 -E11_1 -E11_2" intLanes=":SW_Center_1_0 :SW_Center_1_1 :SW_Center_1_2 :SW_Center_4_0 :SW_Center_11_0 :SW_Center_11_1 :SW_Center_11_2 :SW_Center_14_0 :SW_Center_15_0 :SW_Center_16_0 :SW_Center_16_1 :SW_Center_16_2"/>
    <junction id=":SW_Center_22_0" type="internal" x="-249.11" y="-258.64" incLanes=":SW_Center_14_0 E3_0 E3_1 E3_2" intLanes=":SW_Center_0_0 :SW_Center_1_0 :SW_Center_1_1 :SW_Center_1_2 :SW_Center_6_0 :SW_Center_6_1 :SW_Center_6_2 :SW_Center_9_0 :SW_Center_16_0 :SW_Center_16_1 :SW_Center_16_2 :SW_Center_19_0"/>
    <junction id=":SW_Center_23_0" type="internal" x="-258.64" y="-250.89" incLanes=":SW_Center_19_0 E7_0 E7_1 E7_2" intLanes=":SW_Center_1_0 :SW_Center_1_1 :SW_Center_1_2 :SW_Center_4_0 :SW_Center_5_0 :SW_Center_6_0 :SW_Center_6_1 :SW_Center_6_2 :SW_Center_11_0 :SW_Center_11_1 :SW_Center_11_2 :SW_Center_14_0"/>

    <connection from="-E1" to="-E2" fromLane="0" toLane="0" via=":NW_Center_5_0" tl="NW_Center" linkIndex="5" dir="r" state="o"/>
    <connection from="-E1" to="-E0" fromLane="0" toLane="0" via=":NW_Center_6_0" tl="NW_Center" linkIndex="6" dir="s" state="o"/>
    <connection from="-E1" to="-E0" fromLane="1" toLane="1" via=":NW_Center_6_1" tl="NW_Center" linkIndex="7" dir="s" state="o"/>
    <connection from="-E1" to="-E0" fromLane="2" toLane="2" via=":NW_Center_6_2" tl="NW_Center" linkIndex="8" dir="s" state="o"/>
    <connection from="-E1" to="E3" fromLane="2" toLane="2" via=":NW_Center_9_0" tl="NW_Center" linkIndex="9" dir="l" state="o"/>
    <connection from="-E10" to="-E7" fromLane="0" toLane="0" via=":SW_Center_10_0" tl="SW_Center" linkIndex="10" dir="r" state="O"/>
    <connection from="-E10" to="-E3" fromLane="0" toLane="0" via=":SW_Center_11_0" tl="SW_Center" linkIndex="11" dir="s" state="O"/>
    <connection from="-E10" to="-E3" fromLane="1" toLane="1" via=":SW_Center_11_1" tl="SW_Center" linkIndex="12" dir="s" state="O"/>
    <connection from="-E10" to="-E3" fromLane="2" toLane="2" via=":SW_Center_11_2" tl="SW_Center" linkIndex="13" dir="s" state="O"/>
    <connection from="-E10" to="E11" fromLane="2" toLane="2" via=":SW_Center_14_0" tl="SW_Center" linkIndex="14" dir="l" state="o"/>
    <connection from="-E11" to="E10" fromLane="0" toLane="0" via=":SW_Center_15_0" tl="SW_Center" linkIndex="15" dir="r" state="o"/>
    <connection from="-E11" to="-E7" fromLane="0" toLane="0" via=":SW_Center_16_0" tl="SW_Center" linkIndex="16" dir="s" state="o"/>
    <connection from="-E11" to="-E7" fromLane="1" toLane="1" via=":SW_Center_16_1" tl="SW_Center" linkIndex="17" dir="s" state="o"/>
    <connection from="-E11" to="-E7" fromLane="2" toLane="2" via=":SW_Center_16_2" tl="SW_Center" linkIndex="18" dir="s" state="o"/>
    <connection from="-E11" to="-E3" fromLane="2" toLane="2" via=":SW_Center_19_0" tl="SW_Center" linkIndex="19" dir="l" state="o"/>
    <connection from="-E3" to="E1" fromLane="0" toLane="0" via=":NW_Center_10_0" tl="NW_Center" linkIndex="10" dir="r" state="O"/>
    <connection from="-E3" to="-E2" fromLane="0" toLane="0" via=":NW_Center_11_0" tl="NW_Center" linkIndex="11" dir="s" state="O"/>
    <connection from="-E3" to="-E2" fromLane="1" toLane="1" via=":NW_Center_11_1" tl="NW_Center" linkIndex="12" dir="s" state="O"/>
    <connection from="-E3" to="-E2" fromLane="2" toLane="2" via=":NW_Center_11_2" tl="NW_Center" linkIndex="13" dir="s" state="O"/>
    <connection from="-E3" to="-E0" fromLane="2" toLane="2" via=":NW_Center_14_0" tl="NW_Center" linkIndex="14" dir="l" state="o"/>
    <connection from="-E4" to="E5" fromLane="0" toLane="0" via=":NE_Center_5_0" tl="NE_Center" linkIndex="5" dir="r" state="o"/>
    <connection from="-E4" to="-E1" fromLane="0" toLane="0" via=":NE_Center_6_0" tl="NE_Center" linkIndex="6" dir="s" state="o"/>
    <connection from="-E4" to="-E1" fromLane="1" toLane="1" via=":NE_Center_6_1" tl="NE_Center" linkIndex="7" dir="s" state="o"/>
    <connection from="-E4" to="-E1" fromLane="2" toLane="2" via=":NE_Center_6_2" tl="NE_Center" linkIndex="8" dir="s" state="o"/>
    <connection from="-E4" to="E6" fromLane="2" toLane="2" via=":NE_Center_9_0" tl="NE_Center" linkIndex="9" dir="l" state="o"/>
    <connection from="-E5" to="-E1" fromLane="0" toLane="0" via=":NE_Center_0_0" tl="NE_Center" linkIndex="0" dir="r" state="O"/>
    <connection from="-E5" to="E6" fromLane="0" toLane="0" via=":NE_Center_1_0" tl="NE_Center" linkIndex="1" dir="s" state="O"/>
    <connection from="-E5" to="E6" fromLane="1" toLane="1" via=":NE_Center_1_1" tl="NE_Center" linkIndex="2" dir="s" state="O"/>
    <connection from="-E5" to="E6" fromLane="2" toLane="2" via=":NE_Center_1_2" tl="NE_Center" linkIndex="3" dir="s" state="O"/>
    <connection from="-E5" to="E4" fromLane="2" toLane="2" via=":NE_Center_4_0" tl="NE_Center" linkIndex="4" dir="l" state="o"/>
    <connection from="-E6" to="E4" fromLane="0" toLane="0" via=":NE_Center_10_0" tl="NE_Center" linkIndex="10" dir="r" state="O"/>
    <connection from="-E6" to="E5" fromLane="0" toLane="0" via=":NE_Center_11_0" tl="NE_Center" linkIndex="11" dir="s" state="O"/>
    <connection from="-E6" to="E5" fromLane="1" toLane="1" via=":NE_Center_11_1" tl="NE_Center" linkIndex="12" dir="s" state="O"/>
    <connection from="-E6" to="E5" fromLane="2" toLane="2" via=":NE_Center_11_2" tl="NE_Center" linkIndex="13" dir="s" state="O"/>
    <connection from="-E6" to="-E1" fromLane="2" toLane="2" via=":NE_Center_14_0" tl="NE_Center" linkIndex="14" dir="l" state="o"/>
    <connection from="-E7" to="E9" fromLane="0" toLane="0" via=":SE_Center_15_0" tl="SE_Center" linkIndex="15" dir="r" state="o"/>
    <connection from="-E7" to="E8" fromLane="0" toLane="0" via=":SE_Center_16_0" tl="SE_Center" linkIndex="16" dir="s" state="o"/>
    <connection from="-E7" to="E8" fromLane="1" toLane="1" via=":SE_Center_16_1" tl="SE_Center" linkIndex="17" dir="s" state="o"/>
    <connection from="-E7" to="E8" fromLane="2" toLane="2" via=":SE_Center_16_2" tl="SE_Center" linkIndex="18" dir="s" state="o"/>
    <connection from="-E7" to="-E6" fromLane="2" toLane="2" via=":SE_Center_19_0" tl="SE_Center" linkIndex="19" dir="l" state="o"/>
    <connection from="-E8" to="-E6" fromLane="0" toLane="0" via=":SE_Center_5_0" tl="SE_Center" linkIndex="5" dir="r" state="o"/>
    <connection from="-E8" to="E7" fromLane="0" toLane="0" via=":SE_Center_6_0" tl="SE_Center" linkIndex="6" dir="s" state="o"/>
    <connection from="-E8" to="E7" fromLane="1" toLane="1" via=":SE_Center_6_1" tl="SE_Center" linkIndex="7" dir="s" state="o"/>
    <connection from="-E8" to="E7" fromLane="2" toLane="2" via=":SE_Center_6_2" tl="SE_Center" linkIndex="8" dir="s" state="o"/>
    <connection from="-E8" to="E9" fromLane="2" toLane="2" via=":SE_Center_9_0" tl="SE_Center" linkIndex="9" dir="l" state="o"/>
    <connection from="-E9" to="E8" fromLane="0" toLane="0" via=":SE_Center_10_0" tl="SE_Center" linkIndex="10" dir="r" state="O"/>
    <connection from="-E9" to="-E6" fromLane="0" toLane="0" via=":SE_Center_11_0" tl="SE_Center" linkIndex="11" dir="s" state="O"/>
    <connection from="-E9" to="-E6" fromLane="1" toLane="1" via=":SE_Center_11_1" tl="SE_Center" linkIndex="12" dir="s" state="O"/>
    <connection from="-E9" to="-E6" fromLane="2" toLane="2" via=":SE_Center_11_2" tl="SE_Center" linkIndex="13" dir="s" state="O"/>
    <connection from="-E9" to="E7" fromLane="2" toLane="2" via=":SE_Center_14_0" tl="SE_Center" linkIndex="14" dir="l" state="o"/>
    <connection from="E0" to="E3" fromLane="0" toLane="0" via=":NW_Center_15_0" tl="NW_Center" linkIndex="15" dir="r" state="o"/>
    <connection from="E0" to="E1" fromLane="0" toLane="0" via=":NW_Center_16_0" tl="NW_Center" linkIndex="16" dir="s" state="o"/>
    <connection from="E0" to="E1" fromLane="1" toLane="1" via=":NW_Center_16_1" tl="NW_Center" linkIndex="17" dir="s" state="o"/>
    <connection from="E0" to="E1" fromLane="2" toLane="2" via=":NW_Center_16_2" tl="NW_Center" linkIndex="18" dir="s" state="o"/>
    <connection from="E0" to="-E2" fromLane="2" toLane="2" via=":NW_Center_19_0" tl="NW_Center" linkIndex="19" dir="l" state="o"/>
    <connection from="E1" to="E6" fromLane="0" toLane="0" via=":NE_Center_15_0" tl="NE_Center" linkIndex="15" dir="r" state="o"/>
    <connection from="E1" to="E4" fromLane="0" toLane="0" via=":NE_Center_16_0" tl="NE_Center" linkIndex="16" dir="s" state="o"/>
    <connection from="E1" to="E4" fromLane="1" toLane="1" via=":NE_Center_16_1" tl="NE_Center" linkIndex="17" dir="s" state="o"/>
    <connection from="E1" to="E4" fromLane="2" toLane="2" via=":NE_Center_16_2" tl="NE_Center" linkIndex="18" dir="s" state="o"/>
    <connection from="E1" to="E5" fromLane="2" toLane="2" via=":NE_Center_19_0" tl="NE_Center" linkIndex="19" dir="l" state="o"/>
    <connection from="E2" to="-E0" fromLane="0" toLane="0" via=":NW_Center_0_0" tl="NW_Center" linkIndex="0" dir="r" state="O"/>
    <connection from="E2" to="E3" fromLane="0" toLane="0" via=":NW_Center_1_0" tl="NW_Center" linkIndex="1" dir="s" state="O"/>
    <connection from="E2" to="E3" fromLane="1" toLane="1" via=":NW_Center_1_1" tl="NW_Center" linkIndex="2" dir="s" state="O"/>
    <connection from="E2" to="E3" fromLane="2" toLane="2" via=":NW_Center_1_2" tl="NW_Center" linkIndex="3" dir="s" state="O"/>
    <connection from="E2" to="E1" fromLane="2" toLane="2" via=":NW_Center_4_0" tl="NW_Center" linkIndex="4" dir="l" state="o"/>
    <connection from="E3" to="E11" fromLane="0" toLane="0" via=":SW_Center_0_0" tl="SW_Center" linkIndex="0" dir="r" state="O"/>
    <connection from="E3" to="E10" fromLane="0" toLane="0" via=":SW_Center_1_0" tl="SW_Center" linkIndex="1" dir="s" state="O"/>
    <connection from="E3" to="E10" fromLane="1" toLane="1" via=":SW_Center_1_1" tl="SW_Center" linkIndex="2" dir="s" state="O"/>
    <connection from="E3" to="E10" fromLane="2" toLane="2" via=":SW_Center_1_2" tl="SW_Center" linkIndex="3" dir="s" state="O"/>
    <connection from="E3" to="-E7" fromLane="2" toLane="2" via=":SW_Center_4_0" tl="SW_Center" linkIndex="4" dir="l" state="o"/>
    <connection from="E6" to="E7" fromLane="0" toLane="0" via=":SE_Center_0_0" tl="SE_Center" linkIndex="0" dir="r" state="O"/>
    <connection from="E6" to="E9" fromLane="0" toLane="0" via=":SE_Center_1_0" tl="SE_Center" linkIndex="1" dir="s" state="O"/>
    <connection from="E6" to="E9" fromLane="1" toLane="1" via=":SE_Center_1_1" tl="SE_Center" linkIndex="2" dir="s" state="O"/>
    <connection from="E6" to="E9" fromLane="2" toLane="2" via=":SE_Center_1_2" tl="SE_Center" linkIndex="3" dir="s" state="O"/>
    <connection from="E6" to="E8" fromLane="2" toLane="2" via=":SE_Center_4_0" tl="SE_Center" linkIndex="4" dir="l" state="o"/>
    <connection from="E7" to="-E3" fromLane="0" toLane="0" via=":SW_Center_5_0" tl="SW_Center" linkIndex="5" dir="r" state="o"/>
    <connection from="E7" to="E11" fromLane="0" toLane="0" via=":SW_Center_6_0" tl="SW_Center" linkIndex="6" dir="s" state="o"/>
    <connection from="E7" to="E11" fromLane="1" toLane="1" via=":SW_Center_6_1" tl="SW_Center" linkIndex="7" dir="s" state="o"/>
    <connection from="E7" to="E11" fromLane="2" toLane="2" via=":SW_Center_6_2" tl="SW_Center" linkIndex="8" dir="s" state="o"/>
    <connection from="E7" to="E10" fromLane="2" toLane="2" via=":SW_Center_9_0" tl="SW_Center" linkIndex="9" dir="l" state="o"/>

    <connection from=":NE_Center_0" to="-E1" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":NE_Center_1" to="E6" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":NE_Center_1" to="E6" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":NE_Center_1" to="E6" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":NE_Center_4" to="E4" fromLane="0" toLane="2" via=":NE_Center_20_0" dir="l" state="m"/>
    <connection from=":NE_Center_20" to="E4" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":NE_Center_5" to="E5" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":NE_Center_6" to="-E1" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":NE_Center_6" to="-E1" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":NE_Center_6" to="-E1" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":NE_Center_9" to="E6" fromLane="0" toLane="2" via=":NE_Center_21_0" dir="l" state="m"/>
    <connection from=":NE_Center_21" to="E6" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":NE_Center_10" to="E4" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":NE_Center_11" to="E5" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":NE_Center_11" to="E5" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":NE_Center_11" to="E5" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":NE_Center_14" to="-E1" fromLane="0" toLane="2" via=":NE_Center_22_0" dir="l" state="m"/>
    <connection from=":NE_Center_22" to="-E1" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":NE_Center_15" to="E6" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":NE_Center_16" to="E4" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":NE_Center_16" to="E4" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":NE_Center_16" to="E4" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":NE_Center_19" to="E5" fromLane="0" toLane="2" via=":NE_Center_23_0" dir="l" state="m"/>
    <connection from=":NE_Center_23" to="E5" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":NW_Center_0" to="-E0" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":NW_Center_1" to="E3" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":NW_Center_1" to="E3" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":NW_Center_1" to="E3" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":NW_Center_4" to="E1" fromLane="0" toLane="2" via=":NW_Center_20_0" dir="l" state="m"/>
    <connection from=":NW_Center_20" to="E1" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":NW_Center_5" to="-E2" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":NW_Center_6" to="-E0" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":NW_Center_6" to="-E0" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":NW_Center_6" to="-E0" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":NW_Center_9" to="E3" fromLane="0" toLane="2" via=":NW_Center_21_0" dir="l" state="m"/>
    <connection from=":NW_Center_21" to="E3" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":NW_Center_10" to="E1" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":NW_Center_11" to="-E2" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":NW_Center_11" to="-E2" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":NW_Center_11" to="-E2" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":NW_Center_14" to="-E0" fromLane="0" toLane="2" via=":NW_Center_22_0" dir="l" state="m"/>
    <connection from=":NW_Center_22" to="-E0" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":NW_Center_15" to="E3" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":NW_Center_16" to="E1" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":NW_Center_16" to="E1" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":NW_Center_16" to="E1" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":NW_Center_19" to="-E2" fromLane="0" toLane="2" via=":NW_Center_23_0" dir="l" state="m"/>
    <connection from=":NW_Center_23" to="-E2" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":SE_Center_0" to="E7" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":SE_Center_1" to="E9" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":SE_Center_1" to="E9" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":SE_Center_1" to="E9" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":SE_Center_4" to="E8" fromLane="0" toLane="2" via=":SE_Center_20_0" dir="l" state="m"/>
    <connection from=":SE_Center_20" to="E8" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":SE_Center_5" to="-E6" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":SE_Center_6" to="E7" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":SE_Center_6" to="E7" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":SE_Center_6" to="E7" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":SE_Center_9" to="E9" fromLane="0" toLane="2" via=":SE_Center_21_0" dir="l" state="m"/>
    <connection from=":SE_Center_21" to="E9" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":SE_Center_10" to="E8" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":SE_Center_11" to="-E6" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":SE_Center_11" to="-E6" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":SE_Center_11" to="-E6" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":SE_Center_14" to="E7" fromLane="0" toLane="2" via=":SE_Center_22_0" dir="l" state="m"/>
    <connection from=":SE_Center_22" to="E7" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":SE_Center_15" to="E9" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":SE_Center_16" to="E8" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":SE_Center_16" to="E8" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":SE_Center_16" to="E8" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":SE_Center_19" to="-E6" fromLane="0" toLane="2" via=":SE_Center_23_0" dir="l" state="m"/>
    <connection from=":SE_Center_23" to="-E6" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":SW_Center_0" to="E11" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":SW_Center_1" to="E10" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":SW_Center_1" to="E10" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":SW_Center_1" to="E10" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":SW_Center_4" to="-E7" fromLane="0" toLane="2" via=":SW_Center_20_0" dir="l" state="m"/>
    <connection from=":SW_Center_20" to="-E7" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":SW_Center_5" to="-E3" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":SW_Center_6" to="E11" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":SW_Center_6" to="E11" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":SW_Center_6" to="E11" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":SW_Center_9" to="E10" fromLane="0" toLane="2" via=":SW_Center_21_0" dir="l" state="m"/>
    <connection from=":SW_Center_21" to="E10" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":SW_Center_10" to="-E7" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":SW_Center_11" to="-E3" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":SW_Center_11" to="-E3" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":SW_Center_11" to="-E3" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":SW_Center_14" to="E11" fromLane="0" toLane="2" via=":SW_Center_22_0" dir="l" state="m"/>
    <connection from=":SW_Center_22" to="E11" fromLane="0" toLane="2" dir="l" state="M"/>
    <connection from=":SW_Center_15" to="E10" fromLane="0" toLane="0" dir="r" state="M"/>
    <connection from=":SW_Center_16" to="-E7" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":SW_Center_16" to="-E7" fromLane="1" toLane="1" dir="s" state="M"/>
    <connection from=":SW_Center_16" to="-E7" fromLane="2" toLane="2" dir="s" state="M"/>
    <connection from=":SW_Center_19" to="-E3" fromLane="0" toLane="2" via=":SW_Center_23_0" dir="l" state="m"/>
    <connection from=":SW_Center_23" to="-E3" fromLane="0" toLane="2" dir="l" state="M"/>

</net>""",file=netfile)

def generate_routefile(args, save_dir="./sumo_data"):
    step_length = args.step_length  # the sim interval length
    num_steps = args.num_steps  # number of time steps
    Lambda = args.Lambda  # arrival rate of car flow
    random_factors = args.random_factor  # random factor for car flow
    accel = args.accel  # accelerate of car flow
    decel = args.decel  # decelerate of car flow
    sigma = (
        args.sigma
    )  # imperfection of drivers, which takes value on [0,1], with 0 meaning perfection and 1 meaning imperfection
    carLength = args.carLength  # length of cars
    minGap = args.minGap  # minimum interval between adjacent cars
    maxSpeed = args.maxSpeed  # maxSpeed for cars

    os.makedirs(save_dir, exist_ok=True)
    # import ipdb
    # ipdb.set_trace()
    
    # command = f"python utils/randomTrips.py \
    #     -n sumo_data/road.net.xml \
    #     -r sumo_data/road_Lbd{args.Lambda:.2f}.rou.xml \
    #     -a sumo_data/vtypes.add.xml \
    #     -t \"type='car'\" \
    #     -e {num_steps} --period {1/Lambda} --random-depart --random-factor {random_factor} --random --flows 8"
    # os.system(command)
    
    # return
    random_factors = 1 + args.random_factor * np.random.random(9)
    probability = Lambda / 9 * random_factors/random_factors.mean()
    
    with open(
        os.path.join(save_dir, f"road_Lbd{args.Lambda:.2f}.rou.xml"), "w"
    ) as routes:
        print(
            f"""<?xml version="1.0" encoding="UTF-8"?>

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
    <flow id="flow0" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{min(1,probability[0])}" route="rd_0"/>
    <flow id="flow1" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{min(1,probability[1])}" route="rd_1"/>
    <flow id="flow2" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{min(1,probability[2])}" route="rd_2"/>
    <flow id="flow3" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{min(1,probability[3])}" route="rd_3"/>
    <flow id="flow4" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{min(1,probability[4])}" route="rd_4"/>
    <flow id="flow5" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{min(1,probability[5])}" route="rd_5"/>
    <flow id="flow6" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{min(1,probability[6])}" route="rd_6"/>
    <flow id="flow7" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{min(1,probability[7])}" route="rd_7"/>
    <flow id="flow8" type="typecar" begin="0.00" color="blue" end="{num_steps}" probability="{min(1,probability[8])}" route="rd_8"/>

</routes>
""",
            file=routes,
        )


def generate_addfile(args, save_dir="./sumo_data"):
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
        os.path.join(save_dir, f"vtypes.add.xml"), "w"
    ) as addfile:
        print(
            f"""<?xml version="1.0" encoding="UTF-8"?>

<additional>
    <vType id="car" 
           vClass="passenger"
           maxSpeed="{maxSpeed}" 
           length="{carLength}" 
           minGap="{minGap}" 
           accel="{accel}" 
           decel="{decel}" 
           sigma="{sigma}" />
</additional>
""",
            file=addfile,
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
    generate_addfile(args, save_dir=save_dir)
    generate_routefile(args, save_dir=save_dir)
    generate_netfile(args,save_dir=save_dir)
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
