%%
% Part B_1 _Prj_#1
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
T=zeros(1,1520);
 TEP=zeros(10000,380);
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
fprintf(' \n')
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
 
  disp('7.Start Training SOM ')
  fprintf(' \n')
  
 
x = P;
dimension1 =1;
dimension2 = 19;
net=newsom(P,[dimension1 dimension2], 'hextop','dist');
view(net)
net.trainParam.epochs = 60;
[trainednet,tr] = train(net,x);
 int_f_som=trainednet.IW{1};
 
 %{
 %Mathematical implementation
 alpha0=0.001;
T2=1000;
 myw=rand(10000,19);
 avl=myw;
 for epoch=1:30
     disp('SOM Epoch #')
     disp(epoch)
     alpha=alpha0*exp(-epoch/T2);
     
     for a1=1:1520
         x=P(:,a1);
           z=x-myw;
           ztavan=z.^2;
           zsum=sum(ztavan);
           [~,minnn]=min(abs(zsum));
           minw=z(:,minnn);
           %subxw=x-minw;
           alphabar=alpha*minw;
           minw=minw+alphabar;
           myw(:,minnn)=minw;
     end
 end
%}
 %
 disp('8.Start Training LVQ ')
fprintf(' \n')  

 int_f_som1=int_f_som;
 
 int_lvq=int_f_som1';
 %int_f_som2=myw;
 
 
           lr0=0.5;
           t22=1000;
  for Epoch=1:1000
          disp('LVQ Epoch #')
          disp(Epoch)
          %lr=lr0-0.0011;
        lr=lr0*exp(-Epoch/t22);
      % lr=lr0/Epoch;
     for a2=1:1:1520
           input=P(:,a2);
        expe=T(1,a2);
    y_in=(int_lvq'*input);
    [~, ind]=max(y_in(:));
    error=expe-ind;
    if error~=0
        int_lvq(:,expe)=int_lvq(:,expe)+(lr.*input);
         int_lvq(:,ind)=int_lvq(:,ind)-(lr.*input);
    else
    end
     end
  end
  
    disp( '9. Starting LVQnetwork Test')
fprintf(' \n')
  final_weights=int_lvq;
  Count=0;
 for o=1:380
  G_10=TEP(:,o);
 TT=TET(1,o);
   G1=G_10'*final_weights;
    [~,L_out_10]=max(G1(:));
  %  disp( 'Real Out   ')
   % disp (TT)
   %  disp('  ...  ')
   % disp (L_out)
     if L_out_10==TT
         Count=Count+1;
     end
 end
 fprintf(' \n')
 fprintf('The Number of Correct Classified Datas with LVQ network is %d \n\n  ',Count)
 fprintf('We had 380 Test Data so the Accuracy will be \n\n')
 acc=Count/o*100;
 fprintf('%2.2f %%\n\n',acc)
 
%%
%Part B_2 _Prj_#1
disp('10. Adding Noise to previous Training Datas')
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

  fprintf('11. Using Noisy Datas as Test Datas')
  fprintf(' \n')


Count_10=0;

 for O10=1:1520
  G_10=newTest_data_10(:,O10);
 TT_10=TestLabels(1,O10);
   G1_10=G_10'*final_weights;
    [~,L_out_10]=max(G1_10(:));
     if L_out_10==TT_10
         Count_10=(Count_10)+1;
     end
 end
  fprintf(' \n')
 fprintf('The Number of Correct Classified Datas by LVQ network  with 10%% noise is %d \n\n  ',Count_10)
 fprintf('We had 1520 Test Data so the Accuracy will be \n\n')
 acc_10=Count_10/O10*100;
  fprintf('%2.2f %%\n\n',acc_10)
 
 
Count_20=0;
 for O20=1:1520
  G_20=newTest_data_20(:,O20);
  TT_20=TestLabels(1,O20);
   G1_20=G_20'*final_weights;
    [~,L_out_20]=max(G1_20(:));
     if L_out_20==TT_20
         Count_20=(Count_20)+1;
     end
 end
 fprintf(' \n')
 fprintf('The Number of Correct Classified Datas by LVQ network  with 20%% noise is %d \n\n  ',Count_20)
 fprintf('We had 1520 Test Data so the Accuracy will be \n\n')
 acc_20=Count_20/O20*100;
  fprintf('%2.2f %%\n\n',acc_20)


