require 'torch'
require 'nn'
require 'optim'
local model_utils = require 'util.model_utils'
require 'nngraph'
opt = {
   dataset = 'folder',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 258,
   fineSize = 256,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 1000,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal
   ip= '131.159.40.120',   -- ip
   port = 8000,            -- port
   lambda=0.5,             -- lambda weighting of the 2 gan losses
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then 
    opt.display = false 
else
    display=require 'display'
    display.configure({hostname=opt.ip, port=opt.port})
end

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

function defineD_n_layers(input_nc, output_nc, ndf, n_layers)
        local netD = nn.Sequential()
        
        -- input is (nc) x 256 x 256
        netD:add(nn.SpatialConvolution(input_nc, ndf, 4, 4, 2, 2, 1, 1))
        netD:add(nn.LeakyReLU(0.2, true))
        
        local nf_mult = 1
        local nf_mult_prev = 1
        for n = 1, n_layers-1 do 
            nf_mult_prev = nf_mult
            nf_mult = math.min(2^n,8)
            netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
            netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        end
        
        -- state size: (ndf*M) x N x N
        nf_mult_prev = nf_mult
        nf_mult = math.min(2^n_layers,8)
        netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 1, 1, 1, 1))
        netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        -- state size: (ndf*M*2) x (N-1) x (N-1)
        netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, 4, 4, 1, 1, 1, 1))
        -- state size: 1 x (N-2) x (N-2)
        
        netD:add(nn.Sigmoid())
        -- state size: 1 x (N-2) x (N-2)
        
        return netD
end

function defineG_unet(input_nc, output_nc, ngf)
    local netG = nil
    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1
    
    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1} - nn.JoinTable(2)
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    
    local o1 = d8 - nn.Tanh()
    
    netG = nn.gModule({e1},{o1})
    
    --graph.dot(netG.fg,'netG')
    
    return netG
end



function defineD_basic(input_nc, output_nc, ndf)
    n_layers = 3
    return defineD_n_layers(input_nc, output_nc, ndf, n_layers)
end

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

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
--G.netG1:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
G.netG1:add(SpatialFullConvolution(ngf, ngf, 4, 4, 2, 2, 1, 1))
G.netG1:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 64 x 64
G.netG1:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
G.netG1:add(nn.Tanh())
-- state size: (nc) x 128 x 128

G.netG1:apply(weights_init)

local D={}
D.netD1 = nn.Sequential()

-- input is (nc) x 128 x 128
D.netD1:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
D.netD1:add(nn.LeakyReLU(0.2, true))
-- state is (ndf) x 64 x 64
D.netD1:add(SpatialConvolution(ndf, ndf, 4, 4, 2, 2, 1, 1))
D.netD1:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
D.netD1:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
D.netD1:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
D.netD1:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
D.netD1:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
D.netD1:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
D.netD1:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
D.netD1:add(SpatialConvolution(ndf * 8, 1, 4, 4))
D.netD1:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
D.netD1:add(nn.View(1):setNumInputDims(3))
-- state size: 1

D.netD1:apply(weights_init)

for i=2,4 do
    G['netG' .. i]=G.netG1:clone()
    D['netD' .. i]=D.netD1:clone()
end


local G_combiner=defineG_unet(3,3,ngf)

local D_combiner=defineD_basic(3,3,ndf)


local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateG_combiner = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD_combiner = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
---------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize/2, opt.fineSize/2)
local input_combiner = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG , errD_combiner, errG_combiner
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda(); input_combiner = input_combiner:cuda() ; noise = noise:cuda();  label = label:cuda()

   --if pcall(require, 'cudnn') then
   --   require 'cudnn'
   --   cudnn.benchmark = true
   --   cudnn.convert(netG, cudnn)
   --   cudnn.convert(netD, cudnn)
   --end
   D_combiner:cuda();           G_combiner:cuda();           
   criterion:cuda()
   for k,net in pairs(G) do net:cuda() end
   for k,net in pairs(D) do net:cuda() end
end

local parametersD, gradParametersD = model_utils.combine_all_parameters(D)  --netD:getParameters()
local parametersG, gradParametersG = model_utils.combine_all_parameters(G)  --netG:getParameters()

local parametersD_combiner , gradParametersD_combiner=D_combiner:getParameters()
local parametersG_combiner , gradParametersG_combiner=G_combiner:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = data:getBatch()
   data_tm:stop()
   --input:copy(real)
   label:fill(real_label)
   
   local errD_real=0

   for i=1,2 do
      for j=1,2 do
         input:copy(real[ {{} , { } , {  1 + (i-1)*opt.fineSize/2 , (i-1)*opt.fineSize/2 + opt.fineSize/2  } , {   1 + (j-1)*opt.fineSize/2 , (j-1)*opt.fineSize/2 + opt.fineSize/2 } }  ])
   	     local output = D['netD'.. tostring( (i-1)*2+j ) ]:forward(input)
         errD_real = errD_real + criterion:forward(output, label)
         local df_do = criterion:backward(output, label)
         D['netD'.. tostring( (i-1)*2+j ) ]:backward(input, df_do)
      end
   end
   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end

   label:fill(fake_label)
   local errD_fake=0
   for i=1,2 do
      for j=1,2 do   
         local fake = G['netG' .. tostring( (i-1)*2+j) ]:forward(noise)
         input:copy(fake)
         local output = D['netD'.. tostring( i-1 )*2 + j ] :forward(input)
         errD_fake = errD_fake + criterion:forward(output, label)
         local df_do = criterion:backward(output, label)
         D['netD' .. tostring(i-1)*2+j]:backward(input, df_do)
      end
   end
   errD = errD_real + errD_fake

   return errD, gradParametersD
end

local fDx_combiner=function(x)
	gradParametersD_combiner:zero()
	data_tm:reset(); data_tm:resume()
	local real=data:getBatch()
	data_tm:stop()
	real=real:cuda()
	-- train with real
	local output=D_combiner:forward(real)	
	local label_combiner=output:clone()
	label_combiner:fill(real_label)
	local errD_real_combiner=criterion:forward(output,label_combiner)
	local df_do=criterion:backward(output,label_combiner)
	D_combiner:backward(real,df_do)
	-- train with real

	for i=1,2 do
		for j=1,2 do
			input_combiner[ {{} , { } , {  1 + (i-1)*opt.fineSize/2 , (i-1)*opt.fineSize/2 + opt.fineSize/2  } , {   1 + (j-1)*opt.fineSize/2 , (j-1)*opt.fineSize/2 + opt.fineSize/2 } }  ] = G['netG'.. tostring( (i-1)*2+j ) ].output
		end
	end
	local fake=G_combiner:forward(input_combiner)
	local output=D_combiner:forward(fake)
	label_combiner:fill(fake_label)
	local errD_fake_combiner=criterion:forward(output,label_combiner)
	local df_do=criterion:backward(output,label_combiner)
	D_combiner:backward(fake,df_do)
	errD_combiner=errD_real_combiner+errD_real_combiner	
	
	return errD_combiner,gradParametersD_combiner
end

local df_dg_combiner
local fGx_combiner=function(x)
	gradParametersG_combiner:zero()
	local label_combiner=D_combiner.output:clone()
	label_combiner:fill(real_label)
	
	local output=D_combiner:forward(G_combiner.output)
	errG_combiner=criterion:forward(output,label_combiner)
	local df_do=criterion:backward(output,label_combiner)
	local df_dg=D_combiner:updateGradInput(G_combiner.output,df_do)
		
	df_dg_combiner=G_combiner:backward(input_combiner,df_dg)
	return errG_combiner, gradParametersG_combiner
end


-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   --local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   --errG = criterion:forward(output, label)
   --local df_do = criterion:backward(output, label)
   --local df_dg = netD:updateGradInput(input, df_do)

   --netG:backward(noise, df_dg)
  errG=0
   
   for i=1,2 do
      for j=1,2 do
         local output=D['netD'.. tostring( (i-1)*2+j ) ]:forward(G['netG'.. tostring( (i-1)*2+j ) ].output )
         errG=errG+opt.lambda*criterion:forward(output,label)
         local df_do=criterion:backward(output,label)
         local df_dg= opt.lambda* D['netD'..tostring( (i-1)*2+j )]:updateGradInput(  G['netG'.. tostring( (i-1)*2+j ) ].output , df_do  )
         df_dg=df_dg + (1-opt.lambda)*df_dg_combiner[  {{} , { } , {  1 + (i-1)*opt.fineSize/2 , (i-1)*opt.fineSize/2 + opt.fineSize/2  } , {   1 + (j-1)*opt.fineSize/2 , (j-1)*opt.fineSize/2 + opt.fineSize/2 } }  ]
         G['netG'..tostring( (i-1)*2+j )]:backward(noise,df_dg)
      end	
   end

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

      optim.adam(fDx_combiner, parametersD_combiner, optimStateD_combiner)
      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx_combiner, parametersG_combiner, optimStateG_combiner)
      
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local real = data:getBatch()
          disp.image(real, {win=opt.display_id * 3, title=opt.name})
		  
          for j=1,2 do
             for k=1,2 do
                local fake= G['netG'.. tostring((j-1)*2+k)]:forward(noise_vis)
                input_combiner[    {{} , { } , {  1 + (j-1)*opt.fineSize/2 , (j-1)*opt.fineSize/2 + opt.fineSize/2  } , {   1 + (k-1)*opt.fineSize/2 , (k-1)*opt.fineSize/2 + opt.fineSize/2 } }  ] = fake           
            end
          end
          local fake=G_combiner:forward(input_combiner)
          disp.image(fake,{win=opt.display_id+1,title='final_image'})
          disp.image(input_combiner,{win=opt.display_id+2,title='intermediate_image'})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f Err_G_combiner: %.4f Err_D_combiner: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1 , errG_combiner and errG_combiner or -1, errD_combiner and errD_combiner or -1  ))
      end
   end
   paths.mkdir('jigsaw_checkpoints')
   --parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   --parametersG, gradParametersG = nil, nil
   torch.save('jigsaw_checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', {G=G})
   torch.save('jigsaw_checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', {D=D})
   torch.save('jigsaw_checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G_combiner.t7', G_combiner)
   torch.save('jigsaw_checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D_combiner.t7', D_combiner)
   --parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   --parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
