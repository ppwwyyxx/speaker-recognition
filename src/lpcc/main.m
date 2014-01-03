[x,f,p]=wavread('a.wav');
x=filter([1-0.9375],1,x);

len=160;
y=enframe(x,len,len/2);

[b,c]=size(y)

for n=1:b
 yy=y(n,:);
 p=10;
 A=real(lpc3(yy,p));
 size(lpc1)
 a=lpc2lpcc(lpc1);
 lpcc(n,:)=a;
end
size(lpcc)
