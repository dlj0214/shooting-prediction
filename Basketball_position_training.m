clc;
clear;
data=xlsread('D:\matlab 论文\篮球.xlsx','去头去尾');  %不等长数据换为等长

% T =0.00833
%55个样例，每个样例三列特征
%1：31，32：55

for len=1:55

s=data(:,3*len-1);
ll=length(s);
ss=ll-sum(ismissing(s)); 

x=data(1:ss,3*len-2);
y=data(1:ss,3*len-1);
z=(0:0.00833:0.00833*(ss-1))';

xx=450:20:1600;               %改变xx 的值就是改变X坐标的取值范围，XX的取值范围是450-1600，然后以一定的间隔对其抽样 得到X对应的Y坐标，T时间
z1=interp1(x,z,xx,'pchip');   %对时间插值
plot(x,z,'o',xx,z1,'r'); 

y1=interp1(x,y,xx,'pchip');   %对y插值
plot(x,z,'o',xx,y1,'r');

for j=1:length(z1)
  f1(j,3*len-2)=z1(j);         % f1,f2,f3是对于不同的特征组合
  f1(j,3*len-1)=y1(j);
  f1(j,3*len)=xx(j);
  f2(j,2*len-1)=z1(j);
  f2(j,2*len)=sqrt(xx(j)*xx(j)+y1(j)*y1(j));
  f3(j,2*len-1)=z1(j);
  f3(j,2*len)=y1(j);
  
  f4(j,3*len-2)=z1(j);
 % f4(j,3*len-1)=sqrt(xx(j)*xx(j)+y1(j)*y1(j));
%   f4(j,4*len-2)=xx(j);
  f4(j,3*len-1)=y1(j);
  if j<length(z1)
     f4(j,3*len)=(sqrt((xx(j+1)-xx(j))^2+(y1(j+1)-y1(j))^2))/(z1(j+1)-z1(j));
  else
     f4(j,3*len)=(sqrt((xx(j)-xx(j-1))^2+(y1(j)-y1(j-1))^2))/(z1(j)-z1(j-1));
  end
  
end
clear z1;
clear y1;
end
f1=f1(2:end,:);
f2=f2(2:end,:);
f3=f3(2:end,:);
f4=f4(2:end,:);
 f1 = mapminmax(f1',0,1)';
 f2 = mapminmax(f2',0,1)'; %标准化
 f3 = mapminmax(f3',0,1)';

for i=1:55
    ff1(i,:)=reshape(f1(:,3*i-2:3*i)',[],1);
    ff2(i,:)=reshape(f2(:,2*i-1:2*i)',[],1);
    ff3(i,:)=reshape(f3(:,2*i-1:2*i)',[],1);
    ff4(i,:)=reshape(f4(:,3*i-1:3*i)',[],1);
end

data=ff1;

for m=1:size(data,2)/3   %24,119,48
raw_data=data(1:55,1:m*3);
raw_label=[zeros([31,1]);ones([24,1])];

train_label=raw_label(1:2:end,:);
test_label=raw_label(2:2:end,:);
train_data=raw_data(1:2:end,:);
test_data=raw_data(2:2:end,:);

md1=fitcknn(train_data,train_label,'NumNeighbors',1);
md2=fitcknn(train_data,train_label,'NumNeighbors',2);
md3=fitcknn(train_data,train_label,'NumNeighbors',3); 
md4=fitcecoc(train_data,train_label); 
md5=fitctree(train_data,train_label);

label1 = predict(md1,test_data);
label2= predict(md2,test_data);
label3= predict(md3,test_data);
label4= predict(md4,test_data);
label5 = predict(md5,test_data);

error1=0;error2=0;error3=0;error4=0;
error5=0;
for i=1:length(label1)
  if label1(i)==test_label(i)
         error1=error1+1;
  end
   if label2(i)==test_label(i)
          error2=error2+1;
   end
  if label3(i)==test_label(i)
         error3=error3+1;
  end
  if label4(i)==test_label(i)
         error4=error4+1;
  end
  if label5(i)==test_label(i)
         error5=error5+1;
  end
end

knn(m)=100*error1/length(label1);  %最近临
nb(m)=100*error2/length(label1);   %贝叶斯knn
lda(m)=100*error3/length(label1);  %判别分析knn
svm(m)=100*error4/length(label1);  %向量机
cart(m)=100*error5/length(label1); %分类树
end

set(gcf,'color','w');
subplot(3,2,1)
plot(knn,'-o');ylabel('knn');
subplot(3,2,2)
plot(nb,'-o');ylabel('nb');
subplot(3,2,3)
plot(lda,'-o');ylabel('lda');
subplot(3,2,4)
plot(svm,'-o');ylabel('svm');
subplot(3,2,5)
plot(cart,'-o');ylabel('cart');
