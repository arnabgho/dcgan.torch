require 'torch'
require 'nn'
require 'optim'
--require 'rnn'
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
    name = 'experiment-classify_gen_n_alter',
    noise = 'normal',       -- uniform / normal
    ngen = 2,               -- the number of generators generating images                
    ip='131.159.40.120',     -- the ip for display
    port=8000,		    -- the port for display
    save_freq=5, 	    -- the frequency with which the parameters are saved
    lambda=0.5
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
local nmsg = opt.nmsg
local ngen = opt.ngen
local real_label =ngen+ 1
local fake_labels = torch.linspace(1,ngen,ngen)

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
--G.netG1:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
G.netG1:add(SpatialFullConvolution(ngf * 2, nc, 4, 4, 2, 2, 1, 1))
--G.netG1:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
--G.netG1:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
G.netG1:add(nn.Tanh())
-- state size: (nc) x 64 x 64

G.netG1:apply(weights_init)


for i=2,ngen do
    G['netG' .. i]=G.netG1:clone()
--    G['netI' .. i]=G.netI1:clone('weight','bias','gradWeight','gradBias')
--    G['netM' .. i]=G.netM1:clone('weight','bias','gradWeight','gradBias')
    G['netG' .. i]:apply(weights_init)
--    G['netI' .. i]:apply(weights_init)
--    G['netM' .. i]:apply(weights_init)
end

--G.reducer=nn.Sequential()
--G.reducer:add(nn.SplitTable(1,2))
--G.reducer:add(nn.Sequencer(nn.LSTM( nmsg , nmsg )  ))
--G.reducer:add(nn.SelectTable(-1))

local netD = nn.Sequential()

-- input is (nc) x 64 x 64
--netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
--netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
--netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialConvolution(nc, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, ngen+1, 4, 4))
--netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
--netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)
local netD_patch = nn.Sequential()

-- input is (nc) x 64 x 64
--netD_patch:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
--netD_patch:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
--netD_patch:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD_patch:add(SpatialConvolution(nc, ndf * 2, 4, 4, 2, 2, 1, 1))
netD_patch:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD_patch:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD_patch:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD_patch:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD_patch:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD_patch:add(SpatialConvolution(ndf * 8,1, 4, 4))
netD_patch:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
--netD_patch:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD_patch:apply(weights_init)




local criterion_patch = nn.BCECriterion()
local criterion = nn.CrossEntropyCriterion()
---------------------------------------------------------------------------
optimStateG = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}
optimStateD = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}
optimStateD_patch = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize)
local errD, errD_patch , errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
local noise = torch.Tensor(opt.batchSize, nz , 1, 1)
local noise_cache = torch.Tensor(ngen,opt.batchSize , nz ,1 , 1)
--local prev_message = torch.Tensor(opt.batchSize,nmsg,1,1):normal(0,1)
--local prev_noise_cache = torch.Tensor(ngen,opt.batchSize , nz-nmsg ,1 , 1)
--local prev_mapped_message_cache = torch.Tensor(opt.batchSize,ngen,nmsg)
--local prev_fake_cache = torch.Tensor(ngen,opt.batchSize, 3, opt.fineSize, opt.fineSize)
--local provisional_message_cache = torch.Tensor(ngen,opt.batchSize,nmsg,1,1):normal(0,1)
----------------------------------------------------------------------------
if opt.gpu > 0 then
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    input = input:cuda();  noise = noise:cuda();noise_cache=noise_cache:cuda();  label = label:cuda() ;     --   if pcall(require, 'cudnn') then
    --      require 'cudnn'
    --      cudnn.benchmark = true
    --      --cudnn.convert(netG, cudnn)
    --      for k,net in pairs(G) do cudnn.convert(net,cudnn)  end
    --      cudnn.convert(netD, cudnn)
    --   end
    netD:cuda();           --netG:cuda();     
    netD_patch:cuda()
    criterion:cuda()
    criterion_patch:cuda()
    for k,net in pairs(G) do net:cuda() end
end



local parametersD, gradParametersD = netD:getParameters()
local parametersD_patch, gradParametersD_patch= netD_patch:getParameters()
local parametersG, gradParametersG = model_utils.combine_all_parameters(G)

if opt.display then 
   disp = require 'display' 
   disp.configure({hostname=opt.ip,port=opt.port})
end

local noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

------- Forward Through the netI once initially as base case ----

--for i=1,ngen do
--    G['netI'..i]:forward(prev_fake_cache[i])
--    G['netM'..i]:forward( torch.cat({ provisional_message_cache[i]:reshape(opt.batchSize,nmsg), noise_cache[i]:reshape(opt.batchSize,nz-nmsg ) , message:reshape(opt.batchSize,nmsg)  }  ,2 )  )
--end
--G.reducer:forward(prev_mapped_message_cache)

----------------------------------------------------------------


-- create closure to evaluate f(X) and df/dX of discriminator

local fDx = function(x)
    gradParametersD:zero()
    -- Currently 1 batch of real data and 1 batch each from the generators
    -- Might try a version in which n batches of real data along with 1 batch each from the generators
    -- train with real
    data_tm:reset(); data_tm:resume()
    errD=0
    for i=1,ngen do
        local real = data:getBatch()
        data_tm:stop()
        input:copy(real)
        label:fill(real_label)
    
        local output = netD:forward(input)
        errD =errD+ criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        netD:backward(input, df_do)


        if opt.noise == 'uniform' then 
            noise:uniform(-1,1)
        elseif opt.noise=='normal' then
            noise:normal(0,1)
        end
        noise_cache[i]=noise
        local fake=G['netG'..i]:forward( noise)
        input:copy(fake)
        label:fill(fake_labels[i])

        local output=netD:forward(input)
        errD=errD+criterion:forward(output,label)
        local df_do = criterion:backward(output,label)
        netD:backward(input,df_do)
    end
    return errD,gradParametersD 
end

local real_label_patch=1
local fake_label_patch=0

local fDx_patch = function(x)
    gradParametersD_patch:zero()
    errD_patch=0
    for i=1,opt.ngen do
        local real=data:getBatch() 
        input:copy(real)

        local output = netD_patch:forward(input)
        local label_patch=output:clone()
        label_patch:fill(real_label_patch)
        errD_patch=errD_patch+ criterion_patch:forward(output,label_patch)
        local df_do=criterion_patch:backward(output,label_patch)
        netD_patch:backward(input,df_do)

        --local ids=torch.Tensor(opt.batchSize):random(1,ngen)
        --local fake=input:clone()
        --for i=1,opt.batchSize do
        --    fake[i]=G['netG'.. tostring( ids[i] ) ].output[i]
        --end
        fake=G['netG'..i].output
        input:copy(fake)
        local output = netD_patch:forward(input)
        label_patch:fill(fake_label_patch)
        errD_patch= errD_patch + criterion_patch:forward(output,label_patch)
        local df_do=criterion_patch:backward(output,label_patch)
        netD_patch:backward(input,df_do)
    end
    return errD_patch, gradParametersD_patch
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    gradParametersG:zero()
    label:fill(real_label)
    errG=0
    for i=1,ngen do
        local output = netD:forward(G['netG' .. i].output)
        errG = errG + opt.lambda*criterion:forward(output,label)
        local df_do = criterion:backward(output,label)

        local df_dg = opt.lambda * netD:updateGradInput(G['netG'..i].output,df_do)
        --G['netG'..i]:backward(noise,df_dg)

        local output_patch = netD_patch:forward( G['netG'..i].output )
        local label_patch=output_patch:clone()
        label_patch:fill(real_label_patch)
        errG=errG+ (1-opt.lambda)*criterion_patch:forward(output_patch,label_patch)
        local df_do_patch= criterion_patch:backward(output_patch,label_patch)
        df_dg=df_dg+ (1-opt.lambda)*netD_patch:updateGradInput( G['netG'..i].output,df_do_patch )
        G['netG'..i]:backward(noise,df_dg)
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

        optim.adam(fDx_patch, parametersD_patch, optimStateD_patch)
        -- (2) Update G network: maximize log(D(G(z)))
        optim.adam(fGx, parametersG, optimStateG)

        -- display
        counter = counter + 1
        if counter % 10 == 0 and opt.display then
            local real = data:getBatch()
            disp.image(real,  {win=opt.display_id , title=opt.name})
            for i=1,ngen do
                local fake=G['netG'..i]:forward(noise_vis)
                disp.image(fake, {win=opt.display_id+i,title=opt.name..i })
            end
        end

        -- logging
        if ((i-1) / opt.batchSize) % 1 == 0 then
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
            .. '  Err_G: %.4f  Err_D: %.4f ErrD_patch: %.4f'):format(
            epoch, ((i-1) / opt.batchSize),
            math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
            tm:time().real, data_tm:time().real,
            errG and errG or -1, errD and errD or -1, errD_patch and errD_patch or -1 ))
        end
    end
    if epoch % opt.save_freq==0 then	
       paths.mkdir('patch_checkpoints_classify_gen_n_alter')
       --parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
       --parametersG, gradParametersG = nil, nil
       torch.save('patch_checkpoints_classify_gen_n_alter/' .. opt.name .. '_' .. epoch .. '_net_G.t7', {G=G} )
       torch.save('patch_checkpoints_classify_gen_n_alter/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD )
       torch.save('patch_checkpoints_classify_gen_n_alter/' .. opt.name .. '_' .. epoch .. '_net_D_patch.t7', netD_patch )
    end
    --for i=1,ngen do
	--    local fake=G['netG'..i]:forward(noise_vis)
	--    image.save('checkpoints_classify_gen_n_alter/'.. 'gen_'..i .. '_epoch_'.. epoch .. '.jpg',fake:reshape(3,32*8,32*8))
    --end	


    --parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    --parametersG, gradParametersG = netG:getParameters()
    --parametersG, gradParametersG = model_utils.combine_all_parameters(G)
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
    epoch, opt.niter, epoch_tm:time().real))
end
