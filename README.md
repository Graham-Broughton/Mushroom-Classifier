# **Mushroom Classification**
Hello and welcome to my mushroom classification project. This was my final project for the Data Science Bootcamp at Lighthouse Labs.

## **Table of Contents**
1. [__Overview__](https://github.com/Graham-Broughton/mushroom_classification#overview)

## **Overview**
I have been fascinated by our little fungal friends for many years now. The sheer diversity and reproductive strategies is simply staggering... some mushrooms have 20 000+ mating types! I have a BSc in Microbiology and was very close to doing some genetic stufy instead but chose identification last minute to make sure I did not lose anyone during my presentation. I explored this subject on two levels using very similar methodology: 
- A binary identification model specifically for morels and false morels (_Morchella_ and _Gyromitra_ sp.)
- A model to identify all species I could find labeled images of, that had at least 100 photos of each species. (~700 species and 15 000 images)

After sourcing my images I chose to do a 2-step classification process using YOLOv5 to identify and crop mushrooms present, then a CNN (effecientNetv2L) to do the final classification.
