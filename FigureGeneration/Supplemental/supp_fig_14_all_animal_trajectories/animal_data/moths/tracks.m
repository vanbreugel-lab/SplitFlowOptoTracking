function tracks(trialname)
%trialname='1Oct08L.3';
script_composite = strcat(trialname,'.dat');
composite = dlmread(script_composite,'',1,0);

%if last column of composite is zeros, delete
if composite(:,end)==0,
    composite(:,end)=[];
end

%find gaps
zero_rows=[];
[m,n]=size(composite);
for current_row=1:m,
    if composite(current_row,2)==0 && composite(current_row,3)==0,
     zero_rows=[zero_rows,current_row];
    end    
end

% if gaps found, delete
if ~isempty(zero_rows),
    
%find rows before and after gaps
delete_rows=[];
[p,q]=size(zero_rows);
ingap=0;
for n=1:q,
if ingap==0,
    delete_rows=[delete_rows,zero_rows(n)];
    ingap=1;
else
    if n~=q,
        if zero_rows(n)+1==zero_rows(n+1),
            delete_rows=[delete_rows,zero_rows(n)];
        else
            delete_rows=[delete_rows,zero_rows(n)];
            delete_rows=[delete_rows,zero_rows(n)+1];
            ingap=0;
        end   
    else
        delete_rows=[delete_rows,zero_rows(n)];
        delete_rows=[delete_rows,zero_rows(n)+1];
    end
        
end
end
delete_rows=sort(delete_rows);

%remove gaps and rows before and after
[r,s]=size(delete_rows);
for i=1:s
    composite(delete_rows(i),:) = [];
    delete_rows=delete_rows-1;
end

end

%
xcoor=composite(:,2);
ycoor=composite(:,3);
zcoor=composite(:,4);
clear composite;

hold on
%wind tunnel 
line([-.4 -.4 ],[-.15 -.15 ],[.73 -.27 ],'Marker','.','LineStyle','-')
line([-.4 -.4 ],[.85 .85 ],[.73 -.27 ],'Marker','.','LineStyle','-')
line([-.4 -.4 ],[.85 -.15],[.73 .73],'Marker','.','LineStyle','-')
line([-.4 -.4 ],[.85 -.15],[-.27 -.27],'Marker','.','LineStyle','-')

line([2.9 2.9],[-.15 -.15 ],[.73 -.27 ],'Marker','.','LineStyle','-')
line([2.9 2.9],[.85 .85 ],[.73 -.27 ],'Marker','.','LineStyle','-')
line([2.9 2.9],[.85 -.15],[.73 .73],'Marker','.','LineStyle','-')
line([2.9 2.9],[.85 -.15],[-.27 -.27],'Marker','.','LineStyle','-')

line([-.4 2.9],[-.15 -.15],[-.27 -.27],'Marker','.','LineStyle','-')
line([-.4 2.9],[.85 .85],[.73 .73],'Marker','.','LineStyle','-')
line([-.4 2.9],[-.15 -.15],[.73 .73],'Marker','.','LineStyle','-')
line([-.4 2.9],[.85 .85],[-.27 -.27],'Marker','.','LineStyle','-')

%track
plot3(xcoor,ycoor,zcoor)

%release stand
plot3(1.6,.265,.33,'Marker','O')

% %source from calculating in relation to cal frame
% plot3(.05,.265,.33,'Marker','X')

%source from taking the mean x,y,z values from end positions.
plot3(0.1204,0.265,0.2356,'Marker','X')

hold off
