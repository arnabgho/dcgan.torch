require 'image'
require 'nn'
local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 64,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'line',  -- random / line / linefull1d / linefull
    name = 'generation1',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,           -- Display image: 0 = false, 1 = true
    nz = 100,          
    nmsg=50,
    msginter=1, 	   -- Interpolate on message   
    msg_id=1 
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

local sample_input = torch.randn(2,100,1,1)
noise1 = torch.Tensor(opt.batchSize, opt.nz-opt.nmsg, opt.imsize, opt.imsize)
noise2 = torch.Tensor(opt.batchSize, opt.nz-opt.nmsg, opt.imsize, opt.imsize)
message_G1 = torch.Tensor(opt.batchSize ,opt.nmsg , opt.imsize , opt.imsize)
message_G2 = torch.Tensor(opt.batchSize , opt.nmsg , opt.imsize , opt.imsize)
if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    --`net:cuda()
    --cudnn.convert(net, cudnn)
    noise1 = noise1:cuda()
    noise2 = noise2:cuda()
    sample_input = sample_input:cuda()
    message_G1=message_G1:cuda()
    message_G2=message_G2:cuda()
else
   sample_input = sample_input:float()
   net:float()
end

T = torch.load(opt.net)
G=T.G
-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(G.netG1:get(1)) == 'nn.View' then
    G.netG1:remove(1)
end
if torch.type(G.netG2:get(1)) == 'nn.View' then
    G.netG2:remove(1)
end

--print(net)

if opt.noisetype == 'uniform' then
    noise1:uniform(-1, 1)
    noise2:normal(0,1)
    noiseL = torch.FloatTensor(opt.nz-opt.nmsg):uniform(-1, 1)
    noiseR = torch.FloatTensor(opt.nz-opt.nmsg):normal(0, 1)
elseif opt.noisetype == 'normal' then
    noise1:normal(0, 1)
    noise2:uniform(-1,1)
    noiseL = torch.FloatTensor(opt.nz-opt.nmsg):normal(0,1)
    noiseR = torch.FloatTensor(opt.nz-opt.nmsg):uniform(-1,1)
end

local mess_G1=T.message_G1[opt.msg_id]
local mess_G2=T.message_G2[opt.msg_id]

if opt.noisemode == 'line' then
   -- do a linear interpolation in Z space between point A and point B
   -- each sample in the mini-batch is a point on the line
    line  = torch.linspace(0, 1, opt.batchSize)
    if opt.msginter~=1 then
        print("Noise Interpolation")
    	for i = 1, opt.batchSize do
 	   message_G1[i]=mess_G1
	   message_G2[i]=mess_G2
     	   noise1:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
           noise2:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    	end
    else
        print("Message Interpolation")
	for i = 1, opt.batchSize do
 	   noise1[i]=noiseL
	   noise2[i]=noiseR
     	   message_G1[i]=mess_G1 * line[i] + mess_G2 * (1 - line[i])
           message_G2[i]=mess_G1 * line[i] + mess_G2 * (1 - line[i])
    	end
    end

elseif opt.noisemode == 'linefull1d' then
   -- do a linear interpolation in Z space between point A and point B
   -- however, generate the samples convolutionally, so a giant image is produced
    assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
    noise = noise:narrow(3, 1, 1):clone()
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull' then
   -- just like linefull1d above, but try to do it in 2D
    assert(opt.batchSize == 1, 'for linefull mode, give batchSize(1) and imsize > 1')
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
end


-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
--optnet.optimizeMemory(net, sample_input)


local images_G1 = G.netG1:forward(torch.cat(noise1,message_G2,2))
print('Images size: ', images_G1:size(1)..' x '..images_G1:size(2) ..' x '..images_G1:size(3)..' x '..images_G1:size(4))
images_G1:add(1):mul(0.5)
print('Min, Max, Mean, Stdv', images_G1:min(), images_G1:max(), images_G1:mean(), images_G1:std())
image.save(opt.name .. 'images_G1.png', image.toDisplayTensor(images_G1))
print('Saved image to: ', opt.name .. 'images_G1.png')

local images_G2 = G.netG2:forward(torch.cat(noise2,message_G1,2))
print('Images size: ', images_G2:size(1)..' x '..images_G2:size(2) ..' x '..images_G2:size(3)..' x '..images_G2:size(4))
images_G2:add(1):mul(0.5)
print('Min, Max, Mean, Stdv', images_G2:min(), images_G2:max(), images_G2:mean(), images_G2:std())
image.save(opt.name .. 'images_G2.png', image.toDisplayTensor(images_G2))
print('Saved image to: ', opt.name .. 'images_G2.png')


if opt.display then
    disp = require 'display'
    disp.image(images_G1)
    disp.image(images_G2)
    print('Displayed image')
end
