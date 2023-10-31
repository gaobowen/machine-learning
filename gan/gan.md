


```py
G = nn.Sequential(
    nn.Linear(N_IDEAS,128),
    nn.ReLU(),
    nn.Linear(128,ART_COMPONETS)
)
D = nn.Sequential(
    nn.Linear(ART_COMPONETS,128),
    nn.ReLU(),
    nn.Linear(128,1),
    nn.Sigmoid()
)

optimizer_G = torch.optim.Adam(G.parameters(),lr=LR_G)
optimizer_D = torch.optim.Adam(D.parameters(),lr=LR_D)



plt.ion()

# 防止log时值为0
eps = 1e-6

# 网上资料的损失函数用的是log, 对离群点比较好，但是MSE也可以,MSE运算速度快点
criterion = nn.MSELoss()

# 我们希望真数据判别出来的结果是1，损失函数是 MSE 时用到
real_expected_output = torch.ones((BATCH_SIZE,)).to(device)
# 假数据判别出来的结果希望是0，损失函数是 MSE 时用到
fake_expected_output = torch.zeros((BATCH_SIZE,)).to(device)

for step in range(10000):
#批量产生指导数据
real_data = artist_work().to(device)
#为D网络对于指导数据，评判后的输出结果，肯定越接近1越好
d_real_output = D(real_data)

#即将输入G网络的一组随机数
g_fake_input = torch.randn(BATCH_SIZE, N_IDEAS).to(device)
#G网络根据这组随机数产生一组数据
g_fake_output = G(g_fake_input)
#为D网络对于生成的数据，评判后的输出结果，结果越接近0越好，当然，既然是伪造，也不可能结果真的接近于0，最好是比0.5小点，要不然怎么叫造假
d_fake_output = D(g_fake_output)

# G的任务就是生成的数据越接近真实越好，G的损失函数就是和1作比较，越接近1越好，但是等于1就错了，我们只是让趋势往1方向去
g_loss = -torch.mean(torch.log(1.0 - d_fake_output))
# g_loss = criterion(real_expected_output, d_fake_output)

optimizer_g.zero_grad()
g_loss.backward()
optimizer_g.step()

# D的任务要成功判别出真实数据和伪造数据，不能用 d_fake_output.detach()，会导致G网络得不到有效更新，D应该带着G一起更新
d_loss = -torch.mean(torch.log(eps + 1.0 - d_real_output) + torch.log(eps + D(G(g_fake_input))))
# d_loss = (criterion(real_expected_output, d_real_output) + criterion(fake_expected_output, D(G(g_fake_input)))) / 2

optimizer_d.zero_grad()
d_loss.backward()
optimizer_d.step()

if step % 200 == 0: # plotting
print("g_loss", g_loss)
print("d_loss", d_loss)

plt.cla()
plt.plot(PAINT_POINTS[0], g_fake_output.data.cpu().numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % d_real_output.data.cpu().numpy().mean(), fontdict={'size': 13})
# plt.text(-.5, 2, 'G_loss= %.2f ' % G_loss.data.numpy(), fontdict={'size': 13})

plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.1)

plt.ioff()
plt.show()
```