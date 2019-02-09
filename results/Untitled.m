clear all
s = load('Saliency_magnitude_true.txt');
q = load('Saliency_magnitude_false.txt');

a = s/255;
b = q/255;

d = round(s+q);

imshow(d)
figure
imshow(q)
figure
imshow(s)

