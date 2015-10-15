function [table_assgn, cust_assgn] = sim_crp(S, decayF, alpha, varargin)
% ddCRP Similarity matrix version
% 
% --- Original Version Below ---
% Distance-dependent Chinese Restaurant Processes(CRP) clustering
%   Implementation of "Distnace Depedent Chinese Restaurant Processes, JMLR 2011, Blei et al."
% 
% Usage
%   [ta, ca] = ddcrp(D, 'lgstc', 0.1, 0.1);
%   [ta, ca] = ddcrp(D, 'exp', 1, 0.3);
%   [ta, ca] = ddcrp(D, 'wnd', 0.5, 0.1);
% 
% Input
%   D       - Distance matrix, n x n
%   decayF  - 'wnd' | 'exp' | 'lgstc'
%   alpha   - scale parameter
% 
% Output
% 
% History
%   create - Taewoo Kim (twkim@unist.ac.kr), 07-16-2015
% 

if size(S, 1) ~= size(S, 2)
    fprintf('The column dimension and the row dimension of the matrix S should be same.\n');
    return;
end


% initialize variables
n = size(S, 1); % # of observation.
a = varargin{1}; % the threshold value of decay function.
F = ddcrp_decay(S, decayF, a);

% assignments
cust_assgn = assign_customers_to_each_other(F, n, alpha);
table_assgn = assign_customers_to_tables(cust_assgn, n);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cust_assgn = assign_customers_to_each_other(F, n, alpha)
cust_assgn = zeros(n, 1);

for i = 1:n
    scores = F(i, :);
    scores(i) = [];
    scores = [scores alpha]; % fake score

    u = rand; % sample a number from uniform distribution ranging 0 to 1
    u = u * sum(scores); % multiplying a normalizing constant

    z = 1;
    while u > scores(z)
        u = u - scores(z);
        z = z + 1;
    end

    if z == numel(scores)
        cust_assgn(i) = i;
    else
        cust_assgn(i) = z;
    end
end



function table_assgn = assign_customers_to_tables(cust_assgn, n)

table_assgn = zeros(n, 1);
table_assgn_marked = zeros(n, 1);

table_idx = 1;
table_assgn(1) = table_idx;

prev = 1;
prev_list = [prev];
while true
    next = cust_assgn(prev);
    table_assgn_marked(prev) = Inf;

    % next customer is already assigned.
    if table_assgn_marked(next) == Inf
        % connected to another table
        if numel(find(find(prev_list == next))) == 0
            table_assgn(prev_list) = table_assgn(next);
            table_idx = table_idx - 1;
        end

        % find a customer not assigned to a table yet.
        for i = 1:n
            if table_assgn_marked(i) ~= Inf
                break;
            end

            if i == n & table_assgn_marked(i) == Inf
                return;
            end
        end
        
        table_idx = table_idx + 1;
        table_assgn(i) = table_idx;
        prev = i;
        prev_list = [prev];

        continue;
    end
    
    table_assgn(next) = table_assgn(prev);
    prev = next;
    prev_list = [prev_list prev];
end





function F = ddcrp_decay(S, decayF, a)
F = zeros(size(S, 1), size(S, 2));

if strcmp(decayF, 'wnd')        F(find(S > a)) = 1;
elseif strcmp(decayF, 'exp')    F = exp(a*S);
elseif strcmp(decayF, 'lgstc')  F = exp(S+a)./(1 + exp(S+a));
end
