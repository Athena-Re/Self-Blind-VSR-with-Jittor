import decimal
import os
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import data_utils
from trainer.trainer import Trainer
from loss import kernel_loss
import time


class Trainer_Flow_Video(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_Flow_Video, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer_Flow_Video")
        assert args.n_sequence == 5, \
            "Just support args.n_sequence=5; but get args.n_sequence={}".format(args.n_sequence)

        self.ksize = args.ksize
        self.kernel_loss_boundaries = kernel_loss.BoundariesLoss(k_size=self.ksize)
        self.kernel_loss_sparse = kernel_loss.SparsityLoss()
        self.kernel_loss_center = kernel_loss.CentralizedLoss(k_size=self.ksize, scale_factor=1. / args.scale)
        self.l1_loss = torch.nn.L1Loss()

        self.downdata_psnr_log = []
        self.cycle_psnr_log = []
        self.mid_loss_log = []
        self.cycle_loss_log = []
        self.kloss_boundaries_log = []
        self.kloss_sparse_log = []
        self.kloss_center_log = []

        if args.load != '.':
            mid_logs = torch.load(os.path.join(ckp.dir, 'mid_logs.pt'))
            self.downdata_psnr_log = mid_logs['downdata_psnr_log']
            self.cycle_psnr_log = mid_logs['cycle_psnr_log']
            self.mid_loss_log = mid_logs['mid_loss_log']
            self.cycle_loss_log = mid_logs['cycle_loss_log']
            self.kloss_boundaries_log = mid_logs['kloss_boundaries_log']
            self.kloss_sparse_log = mid_logs['kloss_sparse_log']
            self.kloss_center_log = mid_logs['kloss_center_log']

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = optim.Adam([{"params": self.model.get_model().in_conv.parameters()},
                                {"params": self.model.get_model().extra_feat.parameters()},
                                {"params": self.model.get_model().fusion_conv.parameters()},
                                {"params": self.model.get_model().recons_net.parameters()},
                                {"params": self.model.get_model().upsample_layers.parameters()},
                                {"params": self.model.get_model().out_conv.parameters()},
                                {"params": self.model.get_model().kernel_net.parameters()},
                                {"params": self.model.get_model().flow_net.parameters(), "lr": 1e-6}],
                               **kwargs)
        print(optimizer)
        return optimizer

    def train(self):
        epoch = self.scheduler.last_epoch + 1  # 获取当前epoch
        self.scheduler.step()  # 为下一个epoch更新学习率
        self.loss.step()
        
        print("\n====================================")
        print("开始训练 Epoch {}".format(epoch))
        print("====================================")
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        mid_loss_sum = 0.
        cycle_loss_sum = 0.
        kloss_boundaries_sum = 0.
        kloss_sparse_sum = 0.
        kloss_center_sum = 0.
        train_psnr_sum = 0.
        
        # 获取数据集大小和批次数
        num_batches = len(self.loader_train)
        total_samples = len(self.loader_train.dataset)
        print(f"训练集大小: {total_samples}样本, {num_batches}批次")
        
        # 创建进度条
        train_pbar = tqdm(
            total=num_batches,
            desc=f'训练(Epoch {epoch})',
            ncols=120,
            leave=True,
            position=0,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )

        batch_time_avg = 0
        data_time_avg = 0
        start_time = time.time()
        end_time = time.time()

        for batch, (input, _, kernel, _) in enumerate(self.loader_train):
            # 计算数据加载时间
            data_time = time.time() - end_time
            data_time_avg = (data_time_avg * batch + data_time) / (batch + 1) if batch > 0 else data_time
            
            input = input.to(self.device)

            output_dict, mid_loss = self.model({'x': input, 'mode': 'train'})
            input_center = output_dict['input']
            input_center_cycle = output_dict['input_cycle']
            recons_down = output_dict['recons_down']
            est_kernel = output_dict['est_kernel']

            self.optimizer.zero_grad()

            loss = self.loss(recons_down, input_center)

            cycle_loss = self.l1_loss(input_center_cycle, input_center)
            cycle_loss_sum = cycle_loss_sum + cycle_loss.item()
            loss = loss + cycle_loss

            w1, w2, w3 = 0.5, 0.04, 1

            kloss_boundaries = self.kernel_loss_boundaries(est_kernel)
            kloss_boundaries_sum = kloss_boundaries_sum + kloss_boundaries.item()
            loss = loss + w1 * kloss_boundaries

            kloss_sparse = self.kernel_loss_sparse(est_kernel)
            kloss_sparse_sum = kloss_sparse_sum + kloss_sparse.item()
            loss = loss + w2 * kloss_sparse

            kloss_center = self.kernel_loss_center(est_kernel)
            kloss_center_sum = kloss_center_sum + kloss_center.item()
            loss = loss + w3 * kloss_center

            if mid_loss:  # mid loss is the loss during the model
                loss = loss + self.args.mid_loss_weight * mid_loss
                mid_loss_sum = mid_loss_sum + mid_loss.item()
            
            loss.backward()
            self.optimizer.step()

            self.ckp.report_log(loss.item())
            
            # 计算PSNR用于显示
            with torch.no_grad():
                train_psnr = data_utils.calc_psnr(input_center, recons_down, rgb_range=self.args.rgb_range, is_rgb=True)
                train_psnr_sum = train_psnr_sum + train_psnr
            
            # 更新进度条
            batch_time = time.time() - end_time
            batch_time_avg = (batch_time_avg * batch + batch_time) / (batch + 1) if batch > 0 else batch_time
            end_time = time.time()
            
            # 更新进度条信息 - 只显示关键指标
            postfix_dict = {
                'PSNR': f'{train_psnr:.2f}',
                'loss': f'{loss.item():.3f}',
                'cycle': f'{cycle_loss.item():.3f}'
            }
            train_pbar.set_postfix(**postfix_dict)
            train_pbar.update(1)

            if (batch + 1) % self.args.print_every == 0:
                progress_percent = 100 * (batch + 1) / num_batches
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time / (batch + 1) * num_batches
                estimated_remaining = estimated_total - elapsed_time
                
                self.ckp.write_log('[{}/{} ({:.1f}%)]\t进度: [{}/{}]\t预计剩余时间: {:.1f}分钟\tPSNR: {:.2f}\tLoss: [total: {:.4f}]{}[cycle: {:.4f}][boundaries: {:.4f}][sparse: {:.4f}][center: {:.4f}][mid: {:.4f}]'.format(
                    batch + 1, num_batches, progress_percent,
                    (batch + 1) * self.args.batch_size, total_samples,
                    estimated_remaining / 60,
                    train_psnr_sum / (batch + 1),
                    self.ckp.loss_log[-1] / (batch + 1),
                    self.loss.display_loss(batch),
                    cycle_loss_sum / (batch + 1),
                    kloss_boundaries_sum / (batch + 1),
                    kloss_sparse_sum / (batch + 1),
                    kloss_center_sum / (batch + 1),
                    mid_loss_sum / (batch + 1)
                ))

        # 关闭进度条
        train_pbar.close()
        
        # 显示训练总结
        total_time = time.time() - start_time
        avg_train_psnr = train_psnr_sum / len(self.loader_train)
        self.ckp.write_log("训练Epoch完成: 耗时 {:.2f}秒 ({:.2f}分钟), 平均PSNR: {:.2f}dB".format(
            total_time, total_time / 60, avg_train_psnr
        ))
        
        self.loss.end_log(len(self.loader_train))
        self.mid_loss_log.append(mid_loss_sum / len(self.loader_train))
        self.cycle_loss_log.append(cycle_loss_sum / len(self.loader_train))
        self.kloss_boundaries_log.append(kloss_boundaries_sum / len(self.loader_train))
        self.kloss_sparse_log.append(kloss_sparse_sum / len(self.loader_train))
        self.kloss_center_log.append(kloss_center_sum / len(self.loader_train))
        
        # 训练完成后的保存选项
        print("\n====================================")
        print("训练 Epoch {} 完成！".format(epoch))
        print("选择保存策略:")
        print("1. 立即保存模型（跳过验证）")
        print("2. 先验证再保存（传统方式）")
        print("3. 跳过保存，继续训练")
        print("====================================")
        
        while True:
            choice = input("请输入选择 (1/2/3): ").strip()
            if choice == '1':
                # 立即保存模型
                print("💾 正在立即保存模型...")
                if not self.args.test_only:
                    # 直接保存当前模型状态
                    self.ckp.save(self, epoch, is_best=False)
                    # 保存训练日志
                    torch.save({
                        'downdata_psnr_log': self.downdata_psnr_log,
                        'cycle_psnr_log': self.cycle_psnr_log,
                        'mid_loss_log': self.mid_loss_log,
                        'cycle_loss_log': self.cycle_loss_log,
                        'kloss_boundaries_log': self.kloss_boundaries_log,
                        'kloss_sparse_log': self.kloss_sparse_log,
                        'kloss_center_log': self.kloss_center_log,
                    }, os.path.join(self.ckp.dir, 'mid_logs.pt'))
                    print("✅ 模型已保存 (未验证)")
                return 'save_only'
            elif choice == '2':
                print("🔄 将进行验证后保存...")
                return 'validate_then_save'
            elif choice == '3':
                print("⏭️ 跳过保存，继续训练...")
                return 'skip_save'
            else:
                print("❌ 无效选择，请输入 1、2 或 3")

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\n验证评估:')
        print("\n====================================")
        print("开始验证 Epoch {}".format(epoch))
        print("====================================")
        
        self.model.eval()
        self.ckp.start_log(train=False)
        cycle_psnr_list = []
        downdata_psnr_list = []
        
        test_start_time = time.time()

        with torch.no_grad():
            tqdm_test = tqdm(
                self.loader_test, 
                desc=f'验证(Epoch {epoch})',
                ncols=110,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
            for idx_img, (input, gt, kernel, filename) in enumerate(tqdm_test):

                filename = filename[self.args.n_sequence // 2][0]

                input = input.to(self.device)
                gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)

                output_dict, _ = self.model({'x': input, 'mode': 'val'})
                input_center = output_dict['input']
                input_down_center = output_dict['input_down']
                input_center_cycle = output_dict['input_cycle']
                recons = output_dict['recons']
                recons_down = output_dict['recons_down']
                est_kernel = output_dict['est_kernel']

                cycle_PSNR = data_utils.calc_psnr(input_center, input_center_cycle, rgb_range=self.args.rgb_range, is_rgb=True)
                downdata_PSNR = data_utils.calc_psnr(input_center, recons_down, rgb_range=self.args.rgb_range, is_rgb=True)
                PSNR = data_utils.calc_psnr(gt, recons, rgb_range=self.args.rgb_range, is_rgb=True)
                self.ckp.report_log(PSNR, train=False)
                cycle_psnr_list.append(cycle_PSNR)
                downdata_psnr_list.append(downdata_PSNR)
                
                # 更新进度条
                tqdm_test.set_postfix({
                    'PSNR': f'{PSNR:.2f}',
                    'cycle': f'{cycle_PSNR:.2f}'
                })

                if self.args.save_images:
                    gt, input_center, recons, input_center_cycle, input_down_center, recons_down = data_utils.postprocess(
                        gt, input_center, recons, input_center_cycle, input_down_center, recons_down,
                        rgb_range=self.args.rgb_range,
                        ycbcr_flag=False,
                        device=self.device)

                    gt_kernel = self.auto_crop_kernel(kernel[:, self.args.n_sequence // 2, :, :, :])
                    gt_kernel = self.process_kernel(gt_kernel)

                    est_kernel = self.process_kernel(est_kernel)

                    save_list = [gt, input_center, recons, input_center_cycle,
                                 input_down_center, recons_down, est_kernel, gt_kernel]
                    self.ckp.save_images(filename, save_list, epoch)
            
            # 验证总结
            test_time = time.time() - test_start_time
            
            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\t平均 Cycle-PSNR: {:.3f} Down-PSNR: {:.3f} PSNR: {:.3f} (最佳: {:.3f} @epoch {}) 验证耗时: {:.2f}秒'.format(
                self.args.data_test,
                sum(cycle_psnr_list) / len(cycle_psnr_list),
                sum(downdata_psnr_list) / len(downdata_psnr_list),
                self.ckp.psnr_log[-1],
                best[0], best[1] + 1,
                test_time))
            self.cycle_psnr_log.append(sum(cycle_psnr_list) / len(cycle_psnr_list))
            self.downdata_psnr_log.append(sum(downdata_psnr_list) / len(downdata_psnr_list))
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
                self.ckp.plot_log(self.cycle_psnr_log, filename='cycle_psnr.pdf', title='Cycle PSNR')
                self.ckp.plot_log(self.downdata_psnr_log, filename='downdata_psnr.pdf', title='DownData PSNR')
                self.ckp.plot_log(self.mid_loss_log, filename='mid_loss.pdf', title='Mid Loss')
                self.ckp.plot_log(self.cycle_loss_log, filename='cycle_loss.pdf', title='Cycle Loss')
                self.ckp.plot_log(self.kloss_boundaries_log, filename='kloss_boundaries.pdf', title='Kernel Boundaries Loss')
                self.ckp.plot_log(self.kloss_sparse_log, filename='kloss_sparse.pdf', title='Kernel Sparse Loss')
                self.ckp.plot_log(self.kloss_center_log, filename='kloss_center.pdf', title='Kernel Center Loss')
                torch.save({
                    'downdata_psnr_log': self.downdata_psnr_log,
                    'cycle_psnr_log': self.cycle_psnr_log,
                    'mid_loss_log': self.mid_loss_log,
                    'cycle_loss_log': self.cycle_loss_log,
                    'kloss_boundaries_log': self.kloss_boundaries_log,
                    'kloss_sparse_log': self.kloss_sparse_log,
                    'kloss_center_log': self.kloss_center_log,
                }, os.path.join(self.ckp.dir, 'mid_logs.pt'))

    def auto_crop_kernel(self, kernel):
        end = 0
        for i in range(kernel.size()[2]):
            if kernel[0, 0, end, 0] == -1:
                break
            end += 1
        kernel = kernel[:, :, :end, :end]
        return kernel

    def process_kernel(self, kernel):
        mi = torch.min(kernel)
        ma = torch.max(kernel)
        kernel = (kernel - mi) / (ma - mi)
        kernel = torch.cat([kernel, kernel, kernel], dim=1)
        kernel = kernel.mul(255.).clamp(0, 255).round()
        return kernel
