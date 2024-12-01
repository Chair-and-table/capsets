
\\Generating caps with Pari/GP code https://pari.math.u-bordeaux.fr/


\\List all vectors of F_3^n and permute them, to add randomness.

{
initial(n,r)=V=[];
forvec(v=vector(n,j,[0,2]),V=concat(V,[v]));
s=numtoperm(3^n,if(r,r,random((3^n)!)));
V=vector(3^n,k,V[s[k]]);
return(V)
}

\\Count lines a subset of F_3^n from scratch. For testing purposes.

{
count_lines(p,v)=r=0;
for(j=1,length(v)-1,for(k=j+1,length(v),
if(v[j]!=p&&v[k]!=p&&Mod(1,3)*(p+v[j]+v[k])==Mod(1,3)*vector(length(p),j,0),r++)));
return(r)
}

\\Remove the element p from v.

{
prune(p,v)=w=[];for(j=1,length(v),if(v[j]!=p,w=concat(w,[v[j]])));
return(w)
}

\\If ct is the count of lines through each point of v, this function
\\returns the updated count if p is removed from v.

{
count_update(p,v,ct)=w=[];
for(j=1,length(v),if(v[j]!=p,
w=concat(w,[if(setsearch(Set(v),lift(Mod(2,3)*(p+v[j]))),ct[j]-1,ct[j])])));
return(w)
}

\\initializes V=F_3^n in some order and successively removes a point p of V
\\among those with the maximum number lines through p in V, until it is a cap.

{
run(n)=V=initial(n);C=vector(3^n,j,(3^n-1)/2);
while(1,
if(vecmax(C,&k)==0,break,C=count_update(V[k],V,C);V=prune(V[k],V));
);
return(V)
}

\\variant of run for testing purposes.

{
run_test(n)=V=initial(n);
while(1,
C=vector(length(V),j,count_lines(V[j],V));
r=vecmax(C,&k);
\\print(V);print(C);
if(r==0,break,V=prune(V[k],V))
);
return(V)
}



\\Check if set is a cap.

{
is_cap(C)=r=1;
m=length(C);
for(j=1,m-1,for(k=j+1,m,
if(setsearch(Set(C),lift(Mod(2,3)*(C[j]+C[k]))),r=0;break(2))));
return(r)
}
