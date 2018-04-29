---
title: A Biological Challenge in the Billions
---
*Jiejun Lu, Hongxiang Qiu, Weidong Xu, Zeyu Zhao*

## Overview and Motivation
BASF is the only mass producer of nematodes used for slug protection, utilizing a plant in Littlehampton, UK equipped with 20 vessels that provide over 190,000 L of fermentation capacity and can hold ~ 38 × 10 ^ 12 nematodes. Product quality is continually tested and maintained through refrigeration and efficient shipping logistics to all our customers, which is a key BASF skill. Using chemical pesticides to protect crops and plants from pests can be controversial because those very pesticides can have detrimental consequences. An alternative to chemical pesticides is to encourage naturally occurring organisms in the soil to destroy pests such as slugs, weevils and caterpillars. Nematodes are naturally occurring microscopic worms already present in the soil that actively seek out and destroy pests. Biologicals for pest control is one of the focus areas within BASF’s Agricultural Solutions business. They develop unique formulations of beneficial nematodes and their storage media to provide optimum stability and product performance. The nematodes in the products selectively target problematic insect species, while remaining harmless to beneficial insects (e.g. ladybugs) and nearby wildlife. Finding an efficient way to use nematodes to protect crops could potentially reduce the need to introduce artificial chemicals in to the food supply.

Nematodes have different efficacies at different stages of their life cycles. Automation of identification of whether infective juveniles or not enable efficient quality control. This project applied deep learning techniques to light microscopy image data of nematode populations to determine whether nematodes are at infective juvenile stage or not. We implemented and trained a faster R-CNN model for the identification and classification of nematodes from microscope images, and built a software package in a virtual machine image for the automation of nematode classification task.

## Description of Data and EDA
BASF does not have a digital way to label the nematodes in the microscope images. Initially, we only have some sample microscope images and a slide briefing the characteristics of each life stage of nematodes.

We implemented a labeling tool and delivered it to them. After that, we continuously received data from BASF. Finally, we have 406 valid labeledmicroscopy images, with unbalanced distribution as shown below.

<img src='img/nematode_distribution.png' style='width: 200px;'>












