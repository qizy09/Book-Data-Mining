function [lambdas,y,S,iters] = multisvm2(kernels,labels,C,alpha,lambdasO)
%
% function [lambdas,y,S,iters] = multisvm(kernels,labels,C,alpha,lambdasO)
%
% It is highly recommended to replace the quadprog function
% call with a fast SVM solver (such as SVMperf or Pegasos).
% At the very least please try use MOSEK's quadprog
% function instead of the default Matlab quadprog. 
%
% The lambdasO value is optional and is useful for warm-starts.
% Make sure that it satisfies the SVM constraints ahead of time
% (setting it zero always do this).
%
% THIS VERSION STARTS BOUND AT ZERO BUT ONLY USES LAMBDASO TO SEED QP
%
% Copyright 2009 by Tony Jebara

% Allocate Output Variables
M   = size(labels,2);
D   = size(kernels,2);
S   = ones(D,1);
s   = ones(D,1);
G   = ones(D,1);
Y   = kernels;
T   = zeros(M,1);
for m=1:M
  T(m) = size(labels{m},1);
end

if (nargin<5)
  lambdasO = labels;
  for m=1:M
    lambdasO{m} = zeros(T(m),1);
  end
end;


lambdas = labels;
for m=1:M
  lambdas{m} = zeros(T(m),1);
end


% Compute Predictions and Switches Given Current Lambdas
for d=1:D
  s(d) = 0;
  for m=1:M
    Y{m,d} = kernels{m,d}*(lambdas{m}.*labels{m});
    s(d)   = s(d) + (lambdas{m}.*labels{m})'*kernels{m,d}*(lambdas{m}.*labels{m});
  end
  if (s(d)<(-1000)) s(d)=-1000; end;
  G(d) = alpha*exp(-0.5*s(d));
if (G(d)>0)
  G(d) = tanh(log(G(d))/2)/(2*log(G(d)));
  end
  S(d) = 1/(1+alpha*exp(-0.5*s(d)));
end

% Compute the objective function
J = 0;
for m=1:M
	J = J+sum(lambdas{m});
end
for d=1:D
	J = J - log(alpha*exp(-0.5*s(d))+1)-0.5*s(d);
end
% fprintf('Iteration=0 J=%e\n',J);

going = 1;
iters = 1;
while (going)

  old = lambdas;
qpchange = 0;
for m=1:M
    % Solve the individual QP's
    vlB  = zeros(T(m),1);
    vuB  = C*ones(T(m),1);
    b    = 0;
    A    = labels{m}';
grad = -ones(T(m),1);
for d=1:D
	grad = grad + labels{m}.*(Y{m,d}*S(d));
    end
    hess = zeros(T(m),T(m));
for d=1:D
	hess = hess + G(d)*Y{m,d}*Y{m,d}' + kernels{m,d};
    end
    hess = hess.*(labels{m}*labels{m}');
valOLD = (grad-hess*old{m})'*lambdas{m}+0.5*lambdas{m}'*hess*lambdas{m};
old{m} = lambdas{m};


lambdas{m} = quadprog(hess,grad-hess*lambdas{m},[],[],A,b,vlB,vuB,lambdasO{m});
lambdasO{m} = lambdas{m};
valNEW = (grad-hess*old{m})'*lambdas{m}+0.5*lambdas{m}'*hess*lambdas{m};
qpchange = qpchange+valOLD-valNEW;

% Interlave update for Y, S and G after each model changes...
  for d=1:D
	  Y{m,d} = kernels{m,d}*(lambdas{m}.*labels{m});
s(d)   = s(d) - (lambdas{m}.*labels{m})'*kernels{m,d}*(lambdas{m}.*labels{m});
      s(d)   = s(d) + (old{m}.*old{m})'*kernels{m,d}*(old{m}.*old{m});
if (s(d)<(-1000)) s(d)=-1000; end;
G(d)   = alpha*exp(-0.5*s(d));
if (G(d)>0)
  G(d) = tanh(log(G(d))/2)/(2*log(G(d)));
      end
      S(d) = 1/(1+alpha*exp(-0.5*s(d)));
    end
  end

    % To Be Safe, Cleanly Compute Predictions and Switches Given Current Lambdas
      for d=1:D
	      s(d) = 0;
for m=1:M
	Y{m,d} = kernels{m,d}*(lambdas{m}.*labels{m});
s(d)   = s(d) + (lambdas{m}.*labels{m})'*kernels{m,d}*(lambdas{m}.*labels{m});
    end
    if (s(d)<(-1000)) s(d)=-1000; end;
    G(d) = alpha*exp(-0.5*s(d));
    if (G(d)>0)
      G(d) = tanh(log(G(d))/2)/(2*log(G(d)));
    end
    S(d) = 1/(1+alpha*exp(-0.5*s(d)));
  end


  % Compute the objective function
  J = 0;
  for m=1:M
    J = J+sum(lambdas{m});
  end
  for d=1:D
    J = J - log(alpha*exp(-0.5*s(d))+1)-0.5*s(d);
  end
  
  % Check if lambdas converged
  delt = 0;
  for m=1:M
    delt = max(delt,max(abs(lambdas{m}-old{m})));
  end
  if (delt<(C*1e-3))
    going = 0;
  end

  % Check if number of iterations was exceeded
  if (iters>1000)
    going = 0;
    fprintf('Exceeded max iterations stoping early.\n');
  end
  iters = iters+1;
end

% Finally label the training data
y = labels;
for m=1:M
  y{m} = zeros(T(m),1);
  for d=1:D
    y{m} = y{m}+S(d)*Y{m,d};
  end
end

fprintf('C=%e alpha=%e Iteration=%d J=%e qpimprove=%e\n',C,alpha,iters,J,qpchange);
