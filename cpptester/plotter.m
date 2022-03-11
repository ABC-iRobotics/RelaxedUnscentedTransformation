figure,
subplot(1,2,1),
plotgroundtruth
plotsimresult('Simulator_output_relaxedUT_2','r')
plotsimresult('Simulator_output_relaxedUT_3','g')
plotsimresult('Simulator_output_relaxedUT_4','b')

subplot(1,2,2),
plotgroundtruth
plotsimresult('Simulator_output_UT_2','r')
plotsimresult('Simulator_output_UT_3','g')
plotsimresult('Simulator_output_UT_4','b')
%%
function plotgroundtruth
    groundtruth;
    
    gt_xy = path_groundtruth_t_x_y_phi(1:end,2:3);
    plot(gt_xy(:,1),gt_xy(:,2),'k'); hold on % Groundtruth trajectory
    xy_lm = zeros(length(r),2);
    for i=1:length(r) % plot landmark groundtruth
        xy_lm(i,:) = [r{i}(1),r{i}(2)];
    end
    plot(xy_lm(:,1),xy_lm(:,2),'kx')
    grid on
    xlabel('x[m]'), ylabel('y[m]')
    set(gcf,'renderer','painters'); 
end
%%
function plotsimresult(name,color)
    eval(name)
    
    xy = path__t_x_y_phi(1:end,2:3);
    plot(xy(:,1),xy(:,2),color); % newUT trajectory
    plotxy(x_(1:2),Sx_(1:2,1:2),color);
    % plot landmark uncertainties
    for i=1:(length(x_)-3)/2
        indices = 1+2*i+(1:2);
        plotxy(x_(indices),Sx_(indices,indices),color);
    end
end


function plotxy(xy,Sxy,color)
    plot(xy(1),xy(2),[color 'x']);
    [rx,ry,phi] = cov2ell(Sxy);
    [x,y] = getsegments(xy(1),xy(2),rx,ry,phi);
    plot(x,y,[color '-']);
end

function [x,y] = getsegments(xcenter,ycenter,rx,ry,ang)
    co=cos(ang);
    si=sin(ang);
    the=linspace(0,2*pi,40+1);
    co_the = cos(the);
    si_the = sin(the);
    x=rx*co_the*co-si*ry*si_the+xcenter;
    y=rx*co_the*si+co*ry*si_the+ycenter;
end

function [rx,ry,phi] = cov2ell(S)

    if S(1,1)~=S(2,2)
        phi = atan(2*S(1,2)/(S(1,1)-S(2,2)))/2;

        x = cos(phi)^2 / (cos(phi)^2 - sin(phi)^2);
        y = 1 - x;

        rx = x*S(1,1) + y*S(2,2);
        ry = y*S(1,1) + x*S(2,2);
    else
        phi = pi/4;
        rx = (S(1,1)+S(2,2))/2 + S(1,2);
        ry = (S(1,1)+S(2,2))/2 - S(1,2);
    end
    rx = sqrt(rx);
    ry = sqrt(ry);
end