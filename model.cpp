#include <iostream>
#include <vector>
#include <math.h>
#include "MLP.h"
#include <morph/HdfData.h>
#include <morph/Config.h>
#include "morph/vVector.h"

using namespace std;


int main(int argc, char **argv){
    
    
    std::vector<double> x1;
    std::vector<double> y1;
    std::vector<double> c1;
    morph::HdfData data("img2coded_newd.h5", morph::FileAccess::ReadWrite);
    
    data.read_contained_vals ("x1", x1);
    std::cout<<"x1"<<std::endl;
    data.read_contained_vals ("y1", y1);
    std::cout<<"y1"<<std::endl;
    data.read_contained_vals ("c1", c1);
    std::cout<<"c1"<<std::endl;
    
    std::vector<double> x1_s = x1;
    std::vector<double> y1_s = y1;
    
    int maxx=0;
    for(int i=0;i<x1.size();i++){
    if(x1[i]>maxx){
    maxx=x1[i];
    }
    }
    int maxy= 0;
    for(int i=0;i<y1.size();i++){
    if(y1[i]>maxy){
    maxy=y1[i];
    }
    }
    int maxDim = maxx;
    if(maxy> maxx){
        maxDim = maxy;
    }
    for(int i=0;i<x1.size();i++){
        x1_s[i] = x1[i]/(double)maxDim;
    }
    for(int i=0;i<y1.size();i++){
           y1_s[i] = y1[i]/(double)maxDim;
    }

    
    std::cout << "b x1"<< x1.size()<< std::endl;
    std::cout << "b y1"<< y1.size()<< std::endl;
    std::cout << "b c1"<< c1.size()<< std::endl;
    
    
    // read from a config file (json)
    std::string paramsfile (argv[1]);
    morph::Config conf(paramsfile);
    if (!conf.ready) { std::cerr << "Error setting up JSON config: " << conf.emsg << std::endl; return 1; }

    // create log file
    std::string logpath = argv[2];
    std::ofstream logfile;
    morph::Tools::createDir (logpath);
    { std::stringstream ss; ss << logpath << "/log.txt"; logfile.open(ss.str());}
    logfile<<"Hello."<<std::endl;

    int T = conf.getInt("T", 1000);
    int nHiddenJ = conf.getInt("nHiddenJ", 3);
    int nHiddenQ = conf.getInt("nHiddenQ", 3);
    
    //./model ../config.json logs
    
    //std::cout<<"first element in X1 size"<<X1[0].size()<<std::endl;
    
    // n1 = X1[0]
    // x1 = X1[:,0]
    // y1 = X1[:,1]
    

    
    MLP M(2,nHiddenJ,nHiddenQ,3);
    
    
    /*//XOR problem
    vector<vector<double> > input(4, vector<double> (2, 0.0));
    input[1][1] = 1.0;
    input[2][0] = 1.0;
    input[3][0] = 1.0;
    input[3][1] = 1.0;
    
    vector<double> target(4, 0.0);
    target[1] = 1.0;
    target[2] = 1.0;
     */
    

    
     //
     // One Image Training:

    for (int t = 0; t<T; t++){
        vector<double> input(2, 0.0);
        vector<double> target(3, 0.0);
        
        int r = (int)randDbl(0.0, x1.size());
        input[0] = x1_s[r];
        input[1] = y1_s[r];
        
        
        if (c1[r] > 0.0) {
            target[c1[r]-1] = 1.0; //draw a random x, y color
        }
        
        M.update(0.05, input, target, vector<int> (0)); //no element (0)
    
    }
    
   
    
    // One Image Testing
    vector<double> outNode1(0.0);
    vector<double> outNode2(0.0);
    vector<double> outNode3(0.0);

    
    for (int r = 0; r<x1.size(); r++){
           vector<double> input(2, 0.0);
           vector<double> target(3, 0.0);
           
           input[0] = x1_s[r];
           input[1] = y1_s[r];
           
           M.update(0.00, input, target, vector<int> (0)); //no element (0)
           outNode1.push_back(M.Ks[0]);
           outNode2.push_back(M.Ks[1]);
           outNode3.push_back(M.Ks[2]);
        
       }
    
    
    
    
    //std::cout<<"print Finished training..."<< std::endl;
    //*/
    // end one image
            
    
    //implement context node random draw
    /*
    vector<double> context = randDbl(-1.0, 0.5);//(np.random.rand()<0.5)
    vector<double> target(M.nK*1.0, 0.0)
    std::cout<<“targets initialized to: ”<<target<<std::endl;
    
     if(context){
        r = int(np.floor(np.random.rand() * n1))
        input = [x1[r],y1[r]]
        areaFlag = X1[r,2]
    }
    else{
        r = int(np.floor(np.random.rand() * n2))
        input = [x2[r],y2[r]]
        areaFlag = X2[r,2]
    }
 
     
     if (areaFlag == 1){
         target[2] = 1.0
         
     }
     else if (areaFlag ==2)  {
         target[0] = 1.0}
     else if (areaFlag == 3) {
         target[1] = 1.0
     }
     if((t%100)==0){
         print("progress: ", t/T)
     }
     
     if (context){
         M.update(0.2, input, target, [0])
     }
     else{
         M.update(0.07, input, target, [])
     }
     
    
     */

    
    //XOR Training
    /*
    for (int t = 0; t<T; t++){
       int r = (int)randDbl(0.0, 4.0);
        M.update(0.05, input[r], vector<double> (1, target[r]), vector<int> (0)); //no element (0)
       }
    
    for (int i=0; i<input.size(); i++){
        M.update(0.0, input[i], vector<double> (M.nK, 0.0), vector<int> (0));
        
        std::cout<<"Output "<<M.Ks[0]<<std::endl;
    }
    */
    
    
    // Write data out
    /*
    std::stringstream fname;
    fname << logpath << "/out.h5";
    morph::HdfData data(fname.str());
    std::stringstream path;

    path.str(""); path.clear(); path << "/errors";
    data.add_contained_vals (path.str().c_str(), M.Errors);
     */
    
    //function to scale inputs
    
    
    
    //Save output errors
    {
         std::stringstream fname;
         fname << logpath << "/out.h5";
         morph::HdfData dataout(fname.str());
         std::stringstream path;

         path.str("");path.clear();path << "/errors";
         dataout.add_contained_vals (path.str().c_str(), M.Errors);
        
         path.str("");path.clear();path << "/output1";
         dataout.add_contained_vals (path.str().c_str(), outNode1);
        
         path.str("");path.clear();path << "/output2";
          dataout.add_contained_vals (path.str().c_str(), outNode2);
        
         path.str("");path.clear();path << "/output3";
          dataout.add_contained_vals (path.str().c_str(), outNode3);
        
         path.str("");path.clear();path << "/x1_s";
                 dataout.add_contained_vals (path.str().c_str(), x1_s);
        
         path.str("");path.clear();path << "/y1_s";
                 dataout.add_contained_vals (path.str().c_str(), y1_s);
        
         path.str("");path.clear();path << "/x1";
                 dataout.add_contained_vals (path.str().c_str(), x1);
        
         path.str("");path.clear();path << "/y1";
                 dataout.add_contained_vals (path.str().c_str(), y1);
       }
    
    
    return 0;
   
}





