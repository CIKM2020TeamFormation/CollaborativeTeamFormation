import numpy as np
import random

class TeamFormation:        
    def __init__(self,data):        
        self.dataset=data  
        #self.load_graph()
        #self.save_qraph()
        pfile=open(self.dataset+"/krnmdata1/properties.txt")
        pfile.readline()
        properties=pfile.readline().strip().split(" ")
        pfile.close()
        self.N=int(properties[0]) # number of nodes in the CQA network graph N=|Qestions|+|Answers|+|Experts|                
        self.qnum=int(properties[1])
        self.anum=int(properties[2])
        self.enum=int(properties[3])
        
                 
        self.ccteams=self.loadteams("allcc")
        self.rateams=self.loadteams("allra")
        self.rcoteams=self.loadteams("allrco")
        self.ercoteams=self.loadteams("allerco")
        
        print(self.ccteams)
        print(len(self.rateams))
        print(len(self.rcoteams))
        print(len(self.ercoteams))
        self.averperteam,avrteamsize=self.getaverageteamsize()
        avrteamsize=np.ceil(avrteamsize)
        #self.proposedteams=self.loadproposedteams(avrteamsize)
        #print(self.proposedteams)  
              
        self.testquestions=self.loadtestresults()
        print(len(self.testquestions))
        self.usertags=self.loadusertags() 
        #print(len(self.usertags))
        
        self.questiontags=self.loadquestiontags()
        #print(self.questiontags)
        
               
        #for teamsize in range(3,6,1):
        #    print("\n\n\n\nteam size:"+str(teamsize)) 
        #print("proposed results:")
        #self.displayteams(self.proposedteams)
        print("\ncc results:")
        self.displayteams(self.ccteams)
        print("\nra results:")
        self.displayteams(self.rateams)
        print("\nrco results:")
        self.displayteams(self.rcoteams)   
        print("\nerco results:")
        self.displayteams(self.ercoteams)
        
    def getaverageteamsize(self):
       ccsize=0
       perteam=[]
       for i in range(len(self.ccteams)):
          s1=len(self.ccteams[i])
          s2=len(self.rateams[i])
          s3=len(self.rcoteams[i])
          s4=len(self.ercoteams[i])
          sums=s1+s2+s3+s4
          ccsize+=sums
          perteam.append(int(np.ceil(sums/4)))
       ccsize=ccsize/(4*len(self.ccteams)) 
       return  perteam,ccsize  
    
    def loadproposedteams(self,tsize):
        results=np.loadtxt(self.dataset+"/krnmdata1/DBLPformat/oursigirresults.txt")
        results=results[:,0::2]
        print(results)
        gfile=open(self.dataset+"/krnmdata1/CQAG1.txt")        
        e=gfile.readline()
        G={}
        while e:
            ids=e.split(" ")
            i=int(ids[0])
            j=int(ids[1])
            w=int(ids[2])
            
            if i not in G:
                G[i]={'n':[],'w':[]}
            
            if j not in G:    
                        G[j]={'n':[],'w':[]}
                    
            G[i]['n'].append(j)
            G[i]['w'].append(w) 
            
            G[j]['n'].append(i)
            G[j]['w'].append(w)
            e=gfile.readline()
        N=len(G)
        gfile.close()
        teams=[]
        
        for i in range(len(results)):
            answersid=list(results[i][0:25]) 
            team=[]
            for aid in answersid:
                idmap=self.qnum+aid-1
                for e in G[idmap]['n']:
                    if e>=(self.qnum+self.anum) and e<(self.qnum+self.anum+self.enum):
                       team.append(e)
            
            teams.append(team) 
        return teams 
    def loadteams(self,name):
        filet=open(self.dataset+"/krnmdata1/DBLPformat/"+name+"results.txt")
        
        teams=[]
        line=filet.readline().strip()
        while line:
            eids=line.split(" ")
            team=[]
            for eid in eids:
               team.append(int(eid)+self.qnum+self.anum)
            teams.append(list(set(team)))   
            line=filet.readline().strip()
        return teams
            
    def displayteams(self,teams):
        coverness=0
        total_tags=0
        total_scores=0
        cover_per_q=0
        
        for i in range(len(teams)):
            #print(i)
            qtag=set(self.questiontags[self.testquestions[i][0]]['tags'])
            total_tags+=len(qtag)
            
            #print("\n\nquestion tags=",qtag)
            #print("teams tags:")
            inter=[]
            alltags=[]
            alltags_score=[]
            
            for e in teams[i]:
               #print(e)
               #if (e-self.qnum-self.anum)>self.enum:
               #   continue
               utag=self.usertags[e-self.qnum-self.anum]['tags']
               inter=list(qtag.intersection(utag))
               
               #alltags.extend(inter)
               for t in inter:
                 scor=self.usertags[e-self.qnum-self.anum]['scores'][self.usertags[e-self.qnum-self.anum]['tags'].index(t)] 
                 #print(t," ",scor," ")
                 if t not in alltags:
                     alltags.append(t)
                     alltags_score.append(scor)
                 else:
                     indx=alltags.index(t) 
                     #if alltags_score[indx]<scor:
                     alltags_score[indx]+=(scor)  
            #alltags=list(set(alltags))
            coverness+=len(alltags)
            cover_per_q+= (len(alltags)/len(qtag))
            total_scores+=np.sum(np.array(alltags_score))
            #print(alltags)    
               #print(self.usertags[e-self.qnum-self.anum]['scores'])
            #print("intersection:"+inter)   
        print("total question tags=",total_tags)
        print("total tags covered in teams=",coverness)
        print("total tag scores in teams=",total_scores/coverness)
        print("percent of tags covered in teams=",coverness/total_tags)
        print("percent of tags covered per question=",cover_per_q/len(teams))
        print("average of common quetions answered by team members:",self.findnumcommonquestions(teams))
        print("%.2f & %.2f & %.2f"%((cover_per_q/len(teams))*100,total_scores/coverness,self.findnumcommonquestions(teams)))
    
         
    
    def loadquestiontags(self):
        qtagsfile=open(self.dataset+"/krnmdata1/questionsinfo.txt")
        line=qtagsfile.readline()
        line=qtagsfile.readline().strip()
        qtags={}
        qid=0
        while line:
            token=line.split(" ")           
            tags=token[2:]             
            qtags[qid]={'tags':tags} 
            qid+=1                    
            line=qtagsfile.readline().strip() 
        qtagsfile.close()
        #print(utags)
        return qtags
    
    def loadusertags(self):        
        pufile=open(self.dataset+"/krnmdata1/postusers.txt")
        pufile.readline()
        line=pufile.readline().strip()
        eids=[]
        while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
        pufile.close()
        #print(len(eids))
        
        utagsfile=open(self.dataset+"/krnmdata1/usertags.txt")
        line=utagsfile.readline().strip()
        utags={}
        while line:
            token=line.split(" ")
            eid=int(token[0].strip())  
            if eid in eids:
               tags=token[1::2]
               scores=list(map(int, token[2::2]))
               enewid=eids.index(eid)
               if enewid not in utags: 
                  utags[enewid]={'tags':tags,'scores':scores}
            else:    
                print("error !!! eid is not in eids!! eid="+str(eid))          
            line=utagsfile.readline().strip() 
        utagsfile.close()
        #print(utags)
        return utags
        
        
            
    def loadtestresults(self): 
        testqfile=open(self.dataset+"/krnmdata1/DBLPformat/allccresults.txt") 
        line= testqfile.readline().strip()
        testq=[]
        while line:
             ids=line.split(" ")
             testq.append(int(ids[0]))
             line= testqfile.readline().strip()
        testqfile.close()     
        testqidmap=np.loadtxt(self.dataset+"/krnmdata1/DBLPformat/alltestquestionsids.txt")        
        r=[]
        for i in range(len(testq)):
            r.append([])
            r[i].append(testqidmap[testq[i]]-1)
        return np.array(r)    
       
    def findnumcommonquestions(self,teams):
        gfile=open(self.dataset+"/krnmdata1/CQAG1.txt")        
        e=gfile.readline()
        G={}
        while e:
            ids=e.split(" ")
            i=int(ids[0])
            j=int(ids[1])
            w=int(ids[2])
            
            if i not in G:
                G[i]={'n':[],'w':[]}
            
            if j not in G:    
                        G[j]={'n':[],'w':[]}
                    
            G[i]['n'].append(j)
            G[i]['w'].append(w) 
            
            G[j]['n'].append(i)
            G[j]['w'].append(w)
            e=gfile.readline()
        N=len(G)
        gfile.close()
        
        totalcommonq=0
        for i in range(len(teams)):
             team=teams[i]
             #print(team)
             tq=[]
             for e in team:
                eq=[] 
                for a in G[e]['n']:
                    for q in G[a]['n']:
                       if q!=e:
                          eq.append(q)
                tq.append(eq)
             #print("i=",i,tq)
             commonq=0
             for ii in range(len(tq)):
                qii=set(tq[ii])
                jj=ii+1
                while jj<len(tq):
                   qjj=set(tq[jj])
                   commonq+=len(list(qii.intersection(qjj)))
                   jj+=1
             totalcommonq+= (commonq)     
        return totalcommonq/len(teams)

dataset=["history","dba","english","electronics","softwareengineering","apple"]     
ob=TeamFormation("../data/"+dataset[0])       
        