function lpc1=lpc3(y2,p);
r=zeros(1,p+1);
for k=1:p+1
	sum=0;
	for m=1:160+1-k
		sum=sum+y2(m).*y2(m-1+k);
	end
	r(k)=sum;
end
%¸ù¾ÝdurbinËã·¨ÇóÏßÐÔÔ¤²âÏµÊý
k=zeros(1,p);
k(1)=r(2)/r(1);
a=zeros(p,p);
a(1,1)=k(1);
e=zeros(1,p);
e(1)=(1-k(1)^2)*r(1);
%µÝÍÆ¹ý³Ì
for i=2:p
	c=zeros(1,i);
	sum=0;
	for j=1:i-1
		sum=sum+(a(i-1,j).*r(i+1-j));
	end
	c(i)=sum;
	k(i)=(r(i+1)-c(i))/e(i-1);%Çó·´ÉäÏµÊý
	a(i,i)=k(i);
	for j=1:i-1
		a(i,j)=a(i-1,j)-k(i).*a(i-1,i-j);
	end
	e(i)=(1-k(i)^2)*e(i-1);%ÇóÔ¤²âÎó²î
end
%µÝÍÆ½áÊøºóÌáÈ¡Ô¤²âÏµÊý
d=zeros(1,p);
for t=1:p
	d(t)=a(p,t);
end

lpc1=d;
