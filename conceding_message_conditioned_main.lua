require 'torch'
require 'nn'
require 'optim'
local model_utils = require 'util.model_utils'
opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nmsg = 50,              -- #  of dim for the message
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 20,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment-conceding-message_conditioned1',
   noise = 'normal',       -- uniform / normal
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0
local nmsg = opt.nmsg

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution


local G={}
G.netG1 = nn.Sequential()
-- input is Z, going into a convolution
G.netG1:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
G.netG1:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
G.netG1:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
G.netG1:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
G.netG1:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
G.netG1:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
G.netG1:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
G.netG1:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
G.netG1:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
G.netG1:add(nn.Tanh())
-- state size: (nc) x 64 x 64

G.netG1:apply(weights_init)


G.netG2 = nn.Sequential()
-- input is Z, going into a convolution
G.netG2:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
G.netG2:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
G.netG2:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
G.netG2:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
G.netG2:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
G.netG2:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
G.netG2:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
G.netG2:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
G.netG2:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
G.netG2:add(nn.Tanh())
-- state size: (nc) x 64 x 64

G.netG2:apply(weights_init)

G.netI = nn.Sequential()

-- input is (nc) x 64 x 64
G.netI:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
G.netI:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
G.netI:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
G.netI:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
G.netI:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
G.netI:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
G.netI:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
G.netI:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
G.netI:add(SpatialConvolution(ndf * 8, nmsg, 4, 4))
G.netI:add(SpatialBatchNormalization(nmsg))
--G.netI:add(nn.Sigmoid())
-- state size: nmsg x 1 x 1
--G.netI:add(nn.View(nmsg):setNumInputDims(3))
-- state size: 1

G.netI:apply(weights_init)

G.netI_clone=G.netI:clone('weight','bias','gradWeight','gradBias')

G.netI_clone:apply(weights_init)

G.netM = nn.Sequential()
G.netM:add(nn.Linear((nz-nmsg)+nmsg+nmsg,nmsg))
G.netM:add(nn.BatchNormalization(nmsg))

G.netM:apply(weights_init)

G.netM_clone=G.netM:clone('weight','bias','gradWeight','gradBias')

G.netM_clone:apply(weights_init)

G.relu=nn.ReLU()

local netD = nn.Sequential()

-- input is (nc) x 64 x 64
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)

local criterion = nn.BCECriterion()
local compete_criterion = nn.AbsCriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise1 = torch.Tensor(opt.batchSize, nz - nmsg , 1, 1)
local noise2 = torch.Tensor(opt.batchSize, nz - nmsg , 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
local message_G1 = torch.Tensor(opt.batchSize,nmsg,1,1):normal(0,1)
local message_G2 = torch.Tensor(opt.batchSize,nmsg,1,1):normal(0,1)
local prev_fake1 = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local prev_fake2 = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local provisional_message_G1 = torch.Tensor(opt.batchSize,nmsg,1,1):normal(0,1)
local provisional_message_G2 = torch.Tensor(opt.batchSize,nmsg,1,1):normal(0,1)
local zero_batch=torch.Tensor(opt.batchSize):zero()
local D_G1_out=torch.Tensor(opt.batchSize)
local D_G2_out=torch.Tensor(opt.batchSize)


----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise1 = noise1:cuda();  noise2 = noise2:cuda(); label = label:cuda() ; message_G1=message_G1:cuda() ; message_G2 = message_G2:cuda() ; prev_fake1 = prev_fake1:cuda() ; prev_fake2=prev_fake2:cuda() ; provisional_message_G1=provisional_message_G1:cuda();provisional_message_G2=provisional_message_G2:cuda();zero_batch=zero_batch:cuda();D_G1_out=D_G1_out:cuda();D_G2_out=D_G2_out:cuda()

--   if pcall(require, 'cudnn') then
--      require 'cudnn'
--      cudnn.benchmark = true
--      --cudnn.convert(netG, cudnn)
--      for k,net in pairs(G) do cudnn.convert(net,cudnn)  end
--      cudnn.convert(netD, cudnn)
--   end
   netD:cuda();           --netG:cuda();          
   criterion:cuda()
   compete_criterion:cuda()
   for k,net in pairs(G) do net:cuda() end
end



local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = model_utils.combine_all_parameters(G)

if opt.display then disp = require 'display' end

noise_vis1 = noise1:clone()
noise_vis2 = noise2:clone()
if opt.noise == 'uniform' then
    noise_vis1:uniform(-1, 1)
    noise_vis2:uniform(-1,1)
elseif opt.noise == 'normal' then
    noise_vis1:normal(0, 1)
    noise_vis2:normal(0,1)
end

------- Forward Through the netI once initially as base case ----
G.netI:forward(prev_fake1)
G.netI_clone:forward(prev_fake2)
G.netM:forward( torch.cat({ provisional_message_G1:reshape(opt.batchSize,nmsg), noise1:reshape(opt.batchSize,nz-nmsg ) , message_G2:reshape(opt.batchSize,nmsg)  }  ,2 )  )
G.netM_clone:forward( torch.cat({provisional_message_G2:reshape(opt.batchSize,nmsg),noise2:reshape(opt.batchSize,nz-nmsg ) , message_G1:reshape(opt.batchSize,nmsg)   }  ,2 )  )

----------------------------------------------------------------


-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = data:getBatch()
   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input)
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise1:uniform(-1, 1)
       noise2:uniform(-1,1)
    elseif opt.noise == 'normal' then
       noise1:normal(0, 1)
       noise2:normal(0,1)
   end
   local fake1 = G.netG1:forward(torch.cat(noise1,message_G2,2))
   local fake2 = G.netG2:forward(torch.cat(noise2,message_G1,2))

   input:copy(fake1)
   label:fill(fake_label)

   local output = netD:forward(input)
   local errD_fake1 = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)
   D_G1_out=output   

   input:copy(fake2)
   label:fill(fake_label)

   local output = netD:forward(input)
   local errD_fake2 = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)
   D_G2_out=output

   errD = errD_real + errD_fake1 + errD_fake2

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()
   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   -- assuming all the required outputs have been precomputed in fDx
   label:fill(real_label) -- fake labels are real for generator cost
   --zero_batch:fill(1)
   local diff_G1_G2= D_G2_out-D_G1_out
   local relu_diff_G1_G2=G.relu:forward(diff_G1_G2) 
   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   local errG2 = criterion:forward(output, label) + compete_criterion:forward(relu_diff_G1_G2,zero_batch)
   local df_do = criterion:backward(output, label) + G.relu:backward(diff_G1_G2,compete_criterion:backward(relu_diff_G1_G2,zero_batch))


   local df_dg = netD:updateGradInput(G.netG2.output, df_do)

   local df_input2= G.netG2:backward(torch.cat(noise2,message_G1,2),df_dg)

   local diff_G2_G1= D_G1_out-D_G2_out
   local relu_diff_G2_G1=G.relu:forward(diff_G2_G1)
   local output = netD:forward(G.netG1.output)
   local errG1 = criterion:forward(output,label) + compete_criterion:forward(relu_diff_G2_G1,zero_batch)

   local df_do = criterion:backward(output,label)+ G.relu:backward(diff_G2_G1 , compete_criterion:backward(relu_diff_G2_G1,zero_batch))


   local df_dg = netD:updateGradInput(G.netG1.output,df_do)

   local df_input1=G.netG1:backward(torch.cat(noise1,message_G2,2),df_dg)


   local df_dG2_m1=df_input2[{ {} ,{nz-nmsg+1,nz}}]
   local df_dG1_m2=df_input1[{ {} ,{nz-nmsg+1,nz}}]

   local df_dM1 =  G.netM:backward(  torch.cat({ provisional_message_G1:reshape(opt.batchSize,nmsg), noise1:reshape(opt.batchSize,nz-nmsg ) , message_G2:reshape(opt.batchSize,nmsg)  }  ,2 ) , df_dG2_m1:reshape(opt.batchSize,nmsg))

   local df_dM2 = G.netM_clone:backward( torch.cat({provisional_message_G2:reshape(opt.batchSize,nmsg),noise2:reshape(opt.batchSize,nz-nmsg ) , message_G1:reshape(opt.batchSize,nmsg)   }  ,2 ) , df_dG1_m2:reshape(opt.batchSize,nmsg))

   local df_dI1 = df_dM1[ { {} , { 1 , nmsg  } }]:reshape(opt.batchSize,nmsg,1,1)
   local df_dI2 = df_dM2[ { {} , { 1 , nmsg  } }]:reshape(opt.batchSize,nmsg,1,1)


   G.netI:backward( prev_fake1  , df_dI1 )
   G.netI_clone:backward( prev_fake2 , df_dI2  )

   provisional_message_G1= G.netI:forward(G.netG1.output)
   provisional_message_G2 = G.netI_clone:forward(G.netG2.output)

   local temp_message_G1 = G.netM:forward( torch.cat({ provisional_message_G1:reshape(opt.batchSize,nmsg), noise1:reshape(opt.batchSize,nz-nmsg ) , message_G2:reshape(opt.batchSize,nmsg)  }  ,2 )  )
   local temp_message_G2 = G.netM_clone:forward( torch.cat({provisional_message_G2:reshape(opt.batchSize,nmsg),noise2:reshape(opt.batchSize,nz-nmsg ) , message_G1:reshape(opt.batchSize,nmsg)   }  ,2 )  )

   message_G1=temp_message_G1:reshape(opt.batchSize,nmsg,1,1)
   message_G2=temp_message_G2:reshape(opt.batchSize,nmsg,1,1)
   prev_fake1 = G.netG1.output
   prev_fake2 = G.netG2.output
   errG=errG1+errG2
   return errG, gradParametersG
end


-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local fake1 = G.netG1:forward(torch.cat(noise_vis1,message_G2,2))
          local fake2 = G.netG2:forward(torch.cat(noise_vis2,message_G1,2))
          local real = data:getBatch()
          disp.image(fake1, {win=opt.display_id, title=opt.name})
          disp.image(fake2, {win=opt.display_id * 3, title=opt.name})
          disp.image(real,  {win=opt.display_id*9 , title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints_conceding_message_conditioned')
   --parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   --parametersG, gradParametersG = nil, nil
   torch.save('checkpoints_conceding_message_conditioned/' .. opt.name .. '_' .. epoch .. '_net_G.t7', {G=G,message_G1=message_G1,message_G2=message_G2} )
   torch.save('checkpoints_conceding_message_conditioned/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD )
   --parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   --parametersG, gradParametersG = netG:getParameters()
   --parametersG, gradParametersG = model_utils.combine_all_parameters(G)
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
