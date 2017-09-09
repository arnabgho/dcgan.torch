require 'image'
require 'nn'
--local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 64,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'line',  -- random / line / linefull1d / linefull
    name = '/mnt/raid/arnab/inception',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,           -- Display image: 0 = false, 1 = true
    nz = 100,          
    nmsg=50,
    msginter=1, 	   -- Interpolate on message   
    msg_id=1, 
    ngen=3,
    nsamples=50000,
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

local sample_input = torch.randn(2,100,1,1)
noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
if opt.gpu > 0 then
    require 'cunn'
--    require 'cudnn'
    --`net:cuda()
    --cudnn.convert(net, cudnn)
    cutorch.setDevice(opt.gpu)
    noise = noise:cuda()
    sample_input = sample_input:cuda()
    print("device set")
else
   sample_input = sample_input:float()
   net:float()
end

T = torch.load(opt.net)
G=T.G
-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
for i=1,opt.ngen do
    if torch.type(G['netG'..i]:get(1)) == 'nn.View' then
        G['netG'..i]:remove(1)
    end
end
--print(net)

paths.mkdir(opt.name)
for iter=1,opt.nsamples,opt.batchSize do
    if opt.noisetype == 'uniform' then
        noise:uniform(-1,1)
    elseif opt.noisetype == 'normal' then
        noise:normal(0, 1)
    end
    for i=1,opt.ngen do
        paths.mkdir(opt.name..'/gen_'..tostring(i))
        local images_Gi = G['netG'..i]:forward(noise)
        --print('Images size: ', images_Gi:size(1)..' x '..images_Gi:size(2) ..' x '..images_Gi:size(3)..' x '..images_Gi:size(4))
        images_Gi:add(1):mul(0.5)
        --print('Min, Max, Mean, Stdv', images_G1:min(), images_G1:max(), images_G1:mean(), images_G1:std())

        for j=1,opt.batchSize do
            image.save(opt.name .. '/gen_'..tostring(i).. '/' .. tostring(iter+j-1) ..  '.png', image.toDisplayTensor(images_Gi[j]))
        end
        --print('Saved image to: ', opt.name .. 'images_G1.png')
    end
end
