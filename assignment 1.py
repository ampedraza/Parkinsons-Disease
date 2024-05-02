# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:53:51 2021

@author: aliso
"""

import csv

file = open('diamonds.csv', 'r')
file_reader = csv.reader(file)


rows= []

for row in file_reader:
    rows.append(row)
file.close()
# fliter out the 'Premium' cuts
newlist = list(filter(lambda cut: cut[2]=="Premium", rows))



# Number of entries
print("\nQuestion 1 \n")

print("The number of entries with 'Premium' cut filtered is: \n", len(newlist))


# Compute average weight in carats for Premium group
caratlist = [x[1] for x in newlist]   # list of carat values for Premium
caratfloat = [float(x) for x in caratlist]  # turn strings into floats

weight_t = 0         #sum of carat weight list
for ele in caratfloat:
  weight_t += ele
ave = weight_t / len(caratfloat)
print("The average weight in carats is: \n", ave)


# average price 
pricelist = [int(x[7]) for x in newlist]   # list of prices for Premium
price_t = 0         #sum of pricelist
for x in pricelist:
  price_t += x
aveprice = round(price_t / len(pricelist), 4)
print("\nThe average price for Premium is: ${} \n".format( aveprice))

#p = sum(pricelist)
#print(p)
#print(price_t)

# Question 2 (1,2,3,4,5)
# method a
print("\nQuestion 2 \n")

div_sum = 0
n = 1/13791
divlist = [(price / weight) for price, weight in zip(pricelist, caratfloat)] 

for x in divlist:
    div_sum += x
mean_a = round(div_sum * n, 4)
print("The average price per carat using method 'a' is: ${} \n". format( mean_a))


# method b
mean_b = round(price_t/weight_t,4)
print("The average price per carat using method 'b' is: ${} \n".format(mean_b))

#2
print("Using method 'a', the average price is lower.\n")

#3
# Compute max price per carat
max_p = round(max(divlist), 4)
print("The maximum price per carat: ${}\n".format (max_p))

#4
# Compute minimum price per carat
min_p = round(min(divlist), 4)
print("The minimum price per carat: ${} \n".format( min_p))

#5 Median 
# Compute the median price per carat
divlist.sort()
len_divlist = len(divlist)
median = divlist[len_divlist//2]
print("The median price per carat is: ${}\n". format(median))




# Question 3
#  For each of the two methods to compute price per carat, 
# what combination of other parameters (color, clarity, depth, etc.) gave you 
# highest and lowest price/carat
print("\nQuestion 3 \n")

#ave price using method 'a' and 'b'


# list for ave method a to show corresponding clarity and color
method_a = list(filter(lambda p:p[7] == '4222', newlist))

# list for ave method b to show corresponding clarity and color
method_b = list(filter(lambda p:p[7] == '5139', newlist))

print("1. Higher Value \n")
print("\nThe higher value came from using method 'b'.\n")
print("The color, clarity, and depth combinations using method 'b'  to get" \
      " average price/carat: \n {}\n\n".format(method_b))
    
# using method 'a' - divlist to get the highest price from the list
highest_price_a = round(max(divlist))
lowest_price_a = round(min(divlist))

#print("\nIf using method 'a' only, the highest value of price/carat was: ${}"\
   #   .format(highest_price_a))
#print("The highest price/carat using method 'a' ${}.". format(highest_price_a))
#print("The lowest pricecarat using method 'a' ${}.". format(lowest_price_a))

#get color, clarity, depth combo for highest price using method 'a'
#a = list(filter(lambda p:p[7] == '17083', newlist))
#print("\nThe color, clarity, depth combination with the highest price/carat"\
 #     " from using method 'a' is: \n{}".format(a))

print("2. Lower Value \n")
print("The lower value came from using method 'a': \n")
print("The color, clarity, and depth combination using method 'a' which gave the  "\
      "lower price per carat: \n\n{}\n\n". format(method_a))





highest_price = max(pricelist)
print("\nThe highest priced in Premium diamond from the dataset is: ${}".\
      format(highest_price))
#The highest price/carat using method a

#maxp = list(filter(lambda p:p[7] == '18823', newlist))
#print(list(maxp))

# use pricelist to get the lowest price from the list
lowest_price = min(pricelist)
minp = list(filter(lambda p:p[7]=='326', newlist)) #use newlist to get row
print("The lowest priced Premium diamond from list {}",minp)




# Question 4
#Approximate price if buying a 102 carat diamond using method 'a' for ave price
#per carat
print("\n\nQuestion 4 \n")

price_a = round(mean_a * 102)
print("Using average price per carat using method 'a', the approximate price "
      "I expect to pay for a 102 carat diamond would be: ${} \n". format(price_a))


price_b = round(mean_b * 102)
print("Using average price per carat using method 'b', the approximate price "\
      "I would expect to pay for a 102 carat diamond would be: ${} \n". format(price_b))



