function cap = LCS_ellipse(img,loc)
%% parameters illustration
%1) Tac: 
%The threshold of elliptic angular coverage which ranges from 0~360. 
%The higher Tac, the more complete the detected ellipse should be.
%2) Tr:
%The ratio of support inliers to ellipse which ranges from 0~1.
%The higher Tr, the more sufficient the support inliers are.
%3) specified_polarity: 
%1 means detecting the ellipses with positive polarity;
%-1 means detecting the ellipses with negative polarity; 
%0 means detecting all ellipses from image
%改进功能，在指定矩形范围内识别椭圆
%loc是从detection.txt中提取的某一行
close all;

% parameters
Tac = 160;
Tr = 0.5;
specified_polarity = 0;

%%
% read image 
disp('------read image------');
I = img;
    

%% detecting ellipses from real-world images
[ellipses, ~, posi] = ellipseDetectionByArcSupportLSs(I, Tac, Tr, specified_polarity);


%%
%剔除不在box范围内的椭圆
num=size(ellipses,1);%椭圆个数
r=size(loc,2);
i2dlt=[];%待剔除椭圆索引
if num
    for j=1:num
        i=2;
        flag=0;
        while (true)
            if i>r
                break;
            end
            if loc(i)==0
                break;
            end
            if (loc(i)<ellipses(j,1)<loc(i+2))&&(loc(i+1)<ellispes(j,2)<loc(i+3))
                flag=1;
            end
            i=i+4
        end
        if flag==0
            i2dlt=[i2dlt j];
        end
    end
end
%删除不合格椭圆
if ~isempty(i2dlt)
    
    for i=fliplr(size(i2dlt):1)
        ellipses(i2dlt(i),:)=[];
    end
end
disp('draw detected ellipses');
drawEllipses(ellipses',I);
cap=getframe();
% display
ellipses(:,5) = ellipses(:,5)./pi*180;
ellipses;
disp(['The total number of detected ellipses：',num2str(size(ellipses,1))]);


%% draw ellipse centers
%hold on;
%candidates_xy = round(posi+0.5);%candidates' centers (col_i, row_i)
%plot(candidates_xy(:,1),candidates_xy(:,2),'.');%draw candidates' centers.

%% write the result image
%set(gcf,'position',[0 0 size(I,2) size(I,1)]);
%saveas(gcf, 'D:\Graduate Design\Ellipse Detection\MyEllipse - github\pics\666_all.jpg', 'jpg');
end



