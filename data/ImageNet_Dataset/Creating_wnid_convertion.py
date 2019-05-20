from functions.utils.imagenet_classes import classes

f = open('ImageNet_classes_used.txt', 'r')
wnid = f.read()
wnid = wnid.split()

q = open('map_clsloc.txt', 'r')
contents = q.read()
contents = contents.split()
Indices = []
for i in range(200):
    q = contents.index(wnid[i])
    Class = contents[q+2]
    Class = Class.replace('_', ' ')
    Indices.append(str(Class))
# print(Indices)

True_class = []
for j in range(1000):
    m = classes[j].split(',')[0]
    True_class.append(m)

n = 0
with open('Wnid_convertion.txt', 'w') as w:
    for item in wnid:
        Class_id = True_class.index(Indices[n])
        w.write("{} {}\n".format(item, Class_id))
        n = n + 1