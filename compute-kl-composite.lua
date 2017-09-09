require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'

local opt = lapp[[
   -s,--save          (default "/mnt/raid/arnab/stacked-mnist")      subdirectory to save logs
   -n,--network       (default "/mnt/raid/arnab/stacked-mnist/mnist.net")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 10)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
   --directory        (default "/mnt/raid/arnab/inception") 
   --gen              (default 0)           Use all images by default
]]

-- fix seed
torch.manualSeed(1)

geometry = {32,32}
-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())
model = torch.load(opt.network)

-- verbose
print('<mnist> using model:')
print(model)
local counts=torch.Tensor(1000):zero()
local files=0
local val=0
if opt.gen~=0 then
    for img_name in paths.files() do
        if paths.extname(img_name=='png') then
            img=image.load(opt.directory..'/gen_'+str(opt.gen) .. '/'..img_name)
            val=0
            img=img[1]
            img=img:float()
            for i=0,2 do
                x=i%2
                y=math.floor(i/2)
                _,pred=model:forward(img[{ {1+x*32,x*32+32} , {1+y*32,y*32+32}   } ]:reshape(1,32,32)):max(1)
                val=val*10+(pred[1]%10)
            end
            val=val+1
            counts[val]=counts[val]+1
            files=files+1
            print(files)
        end
    end
else
    for dir in paths.iterdirs(opt.directory) do
        print(dir)
        local dir_count=0
        for img_name in paths.files(opt.directory..'/'..dir) do
            if paths.extname(img_name)=='png' then
                img=image.load(opt.directory..'/'..dir..'/'..img_name)
                val=0
                img=img[3]
                img=img:float()
                for i=0,2 do
                    x=i%2
                    y=math.floor(i/2)
                    _,pred=model:forward(img[{ {1+x*32,x*32+32} , {1+y*32,y*32+32}   } ]:reshape(1,32,32)):max(1)
                    val=val*10+(pred[1]%10)
                    print(pred)
                end
                val=val+1
                counts[val]=counts[val]+1
                files=files+1
                dir_count=dir_count+1
                print(files)
            end
            if opt.full and dir_count==5000 then break end
        end
    end
end

local probs=counts/counts:sum()
kl=0
uniform=1/1000
num_non_zero=0
for i=1,1000 do
    if counts[i]>1e-3 then
        num_non_zero=num_non_zero+1
        kl=kl+probs[i]*torch.log(probs[i]/uniform)
    end
end
print(counts)
print("KL Divergence:")
print(kl)
print("Number of modes covered:")
print(num_non_zero)
