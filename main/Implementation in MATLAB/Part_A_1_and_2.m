%%
% Part A_1 _Prj_#1
%Farshid Pirbonyeh 40033608

clc
clear
close all
%Importing images addresses
disp('1. Importing images addresses')
fprintf(' \n')
imds = imageDatastore("C:\Users\farshid\Desktop\TERM2\Deep\Dataset",...
    "IncludeSubfolders",true,"LabelSource","foldernames","FileExtensions",[".jpg",".png"]);
%Importing Labels
disp('2.Importing Labels')
fprintf(' \n')
T2 = imds.Labels;
Labels= grp2idx(T2);
%getting images with their labels , changing class to uint8 and resizing
disp('3. Getting images with their labels')
fprintf(' \n')
for k=1:1900
    my_field = strcat('v',num2str(k));
    z=imds.Files(k,1);
    z=char(z);
     myLabel = strcat('L',num2str(k));
        o=Labels(k,1);
         Label.(myLabel)=o;
    % variable.Label=Labels(k,1);
    variable.(my_field) = imread(z);
    variable.(my_field) =rgb2gray(variable.(my_field));
     %variable.(my_field)=imbinarize(variable.(my_field));
    variable.(my_field) =imresize(variable.(my_field),[100 100]);
end

la = struct2table( Label );
LA=la{:,:};
%{
disp(' Bipolar labels ')
%Binary labels
TrainTarget1=(zeros(19,1900));%-1;
for j=1:19
   for i=1:1900
    t=LA(1,i);
    if t==j
        TrainTarget1(j,i)=1;
    end
   end
end
%}
ALL_Images=zeros(10000,1900);
ALL_Classes= LA;
for u=1:1900
        my_field = strcat('v',num2str(u));
         s=double(variable.(my_field));
         temp = reshape(s',100*100,1);
         ALL_Images(:,u)=temp;
end         

disp('4. Dividing the dataset to 80% and 20%  ')
fprintf(' \n')
P=zeros(10000,1520);
 %should be use with binary or bipolar labels
% T=zeros(19,1520);
T=zeros(1,1520);
 TEP=zeros(10000,380);
  %should be use with binary or bipolar labels
%TET=zeros(19,380);
TET=zeros(1,380);
  r1=80;
  z1=0;

  m2=zeros(1,1520);
  m1=zeros(1,1520);
for i=1:19
    for j=1:20
          v1=z1+j;
         l1=j+r1;
          %should be use with binary or bipolar labels
          %TET(:,v1)=TrainTarget1(:,l1); 
          TET(1,v1)=LA(1,l1); 
            my_field = strcat('v',num2str(l1));
          s=double(variable.(my_field));
          temp = reshape(s',100*100,1);
          TEP(:,v1)=temp;
            m1(1,v1)=l1;
            m2(1,l1)=v1;
    end
       r1=r1+80;
    z1=z1+20;
end
  m3=zeros(1,380);
  m4=zeros(1,380);
r=0;
z=0;
for i=1:19
    for j=1:80
        v=z+j;
         l=j+r;
          %should be use with binary or bipolar labels
         %T(:,v)=TrainTarget1(:,l); 
         T(1,v)=LA(1,l); 
            my_field = strcat('v',num2str(l));
          s=double(variable.(my_field));
          temp = reshape(s',100*100,1);
          P(:,v)=temp;
          m3(1,v)=l;
          m4(1,l)=v;
    end
    r=r+100;
    z=z+80;
end

disp( '5.Normalizing Datas ')

P=P/255;
TEP=TEP/255;
 
disp( '6.Randomizing Train Data')
fprintf(' \n')
%Randomizing
J=1:1520;
randr = J(randperm(length(J)));
point=zeros(10000,1520);
 target=zeros(1,1520);
 for r=1:1520
     K3=randr(1,r);
     point(:,r)=P(:,K3);
    target(1,r)=T(1,K3);
 end
 P=point;
 T=target;



disp('7.Start Training ')
fprintf(' \n')
%weights=(sqrt(2))/2*rand(19,10000);
weights=zeros(19,10000);
%beta=(.2.*ones(19,1));
beta=0;
alpha=0.135;
theta=0.37;
w_init=weights;
delta_wi=2.*(ones(19,1000));
%while abs(delta_wi(:))> 1.e-25
for epoch=1:300
   epoch_num=epoch;
    disp('# of Epoch is')
    disp(epoch_num)
    disp(' Please Wait ...  ')
     for a1=1:1520
    x=P(:,a1);
    t_expe=T(1,a1);
    y_in=(weights*x);%+beta;
    [maximum, ind]=max(y_in(:));
    %disp( 't_expe  is ')
    %disp (t_expe)
    %disp (ind)
    error=t_expe-ind;
    if error~=0
        weights(t_expe,:)=weights(t_expe,:)+x';
         weights(ind,:)=weights(ind,:)-x';
    else
    end
     end
    % disp( 't_expe   ')
   % disp (t_expe)
   %  disp('  ...  ')
   % disp (ind)
end

disp( '8. Starting Test')
fprintf(' \n')
Count=0;
 for j=1:380
  G_10=TEP(:,j);
 TT=TET(1,j);
   G1=weights*G_10;
    [MO,L_out_10]=max(G1(:));
  %  disp( 'Real Out   ')
   % disp (TT)
   %  disp('  ...  ')
   % disp (L_out)
     if L_out_10==TT
         Count=Count+1;
     end
 end
 fprintf(' \n')
 fprintf('The Number of Correct Classified Datas is %d \n\n  ',Count)
 fprintf('We had 380 Test Data so the Accuracy will be \n\n')
 acc=Count/j*100;
 fprintf('%2.2f %%\n\n',acc)
 
%%
%Part A_2 _Prj_#1
disp('\n9. Adding Noise to previous Training Datas')
 fprintf(' \n')
newTest_data_10=zeros(10000,1520);
TestLabels=zeros(10000,1520);
newTest_data_20=zeros(10000,1520);

r=0;
z=0;
for i=1:19
    for j=1:80
        v=z+j;
         l=j+r;
         
         TestLabels(1,v)=LA(1,l); 
            my_field = strcat('v',num2str(l));
             img=(variable.(my_field));
             img_10=imnoise(img,'salt & pepper',0.1);
             img_20=imnoise(img,'salt & pepper',0.2);
            
          s_10=double(img_10);
          temp_10 = reshape(s_10',100*100,1);
          newTest_data_10(:,v)=temp_10;
         
          
           s_20=double(img_20);
          temp_20 = reshape(s_20',100*100,1);
          
          newTest_data_20(:,v)=temp_20;
         
    end
    r=r+100;
    z=z+80;
end

 fprintf('10. Using Noisy Datas as Test Datas')
  fprintf(' \n')


Count_10=0;

 for j=1:1520
  G_10=newTest_data_10(:,j);
 TT_10=TestLabels(1,j);
   G1_10=weights*G_10;
    [MO,L_out_10]=max(G1_10(:));
  %  disp( 'Real Out   ')
   % disp (TT)
   %  disp('  ...  ')
   % disp (L_out)
     if L_out_10==TT_10
         Count_10=(Count_10)+1;
     end
 end
  fprintf(' \n')
 fprintf('The Number of Correct Classified Datas with 10%% noise is %d \n\n  ',Count_10)
 fprintf('We had 1520 Test Data so the Accuracy will be \n\n')
 acc_10=Count_10/j*100;
  fprintf('%2.2f %%\n\n',acc_10)
 
 
Count_20=0;
 for j=1:1520
  G_20=newTest_data_20(:,j);
  TT_20=TestLabels(1,j);
   G1_20=weights*G_20;
    [MO,L_out_20]=max(G1_20(:));
  %  disp( 'Real Out   ')
   % disp (TT)
   %  disp('  ...  ')
   % disp (L_out)
     if L_out_20==TT_20
         Count_20=(Count_20)+1;
     end
 end
 fprintf(' \n')
 fprintf('The Number of Correct Classified Datas with 20%% noise is %d \n\n  ',Count_20)
 fprintf('We had 1520 Test Data so the Accuracy will be \n\n')
 acc_20=Count_20/j*100;
  fprintf('%2.2f %%\n\n',acc_20)













