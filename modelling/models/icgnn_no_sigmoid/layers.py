import torch

from torch import nn
from utils import get_conv_out_size


class DilatedCausalConv2d(nn.Module):
    def __init__(
        self,
        init_dilation,
        in_channels,
        in_residual_channels,
        out_skip_channels,
        out_residual_channels,
        kernel_size,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.cur_dilation = init_dilation * 2

        self.filter_conv = nn.Conv2d(
            in_channels,
            in_residual_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, self.cur_dilation),
        )
        self.gate_conv = nn.Conv2d(
            in_channels,
            in_residual_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, self.cur_dilation),
        )
        self.skip_conv = nn.Conv2d(
            in_residual_channels, out_skip_channels, kernel_size=(1, 1)
        )
        self.residual_conv = nn.Conv2d(
            in_residual_channels, out_residual_channels, kernel_size=(1, 1)
        )

    def forward(self, x, prev_skip=None):
        """
        x: B x c_in x n x T
        prev_skip: B x c_in x n x T or None
        """

        filter_out = self.filter_conv(x)
        gate_out = self.gate_conv(x)

        filter_out = torch.tanh(filter_out)
        gate_out = torch.sigmoid(gate_out)

        gate_filter_out = filter_out * gate_out

        skip_in = self.skip_conv(gate_filter_out)
        if prev_skip is not None:
            prev_skip = prev_skip[..., -skip_in.size(3) :]
            skip_out = skip_in + prev_skip
        else:
            skip_out = skip_in

        res_in = self.residual_conv(gate_filter_out)
        res_out = res_in + x[..., self.cur_dilation * (self.kernel_size - 1) :]

        return res_out, skip_out


class FDConv2d(nn.Module):
    def __init__(
        self,
        in_ts_length,
        out_ts_length,
        num_layers,
        kernel_size,
        in_residual_channels,
        out_residual_channels,
        out_skip_channels,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        self.dilated_convs = []
        self.dilated_tconvs = []
        cur_dilation = 1
        for i in range(self.num_layers):
            if i == 0:
                self.dilated_convs.append(
                    DilatedCausalConv2d(
                        cur_dilation,
                        in_residual_channels,
                        in_residual_channels,
                        out_skip_channels,
                        out_residual_channels,
                        kernel_size,
                    )
                )
                self.dilated_tconvs.append(
                    nn.ConvTranspose2d(
                        out_skip_channels,
                        in_residual_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, cur_dilation * 2),
                    )
                )
            else:
                self.dilated_convs.append(
                    DilatedCausalConv2d(
                        cur_dilation,
                        out_residual_channels,
                        in_residual_channels,
                        out_skip_channels,
                        out_residual_channels,
                        kernel_size,
                    )
                )
                self.dilated_tconvs.append(
                    nn.ConvTranspose2d(
                        out_skip_channels,
                        out_residual_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, cur_dilation * 2),
                    )
                )
            cur_dilation *= 2

        self.dilated_convs = nn.ModuleList(self.dilated_convs)
        self.dilated_tconvs = nn.ModuleList(self.dilated_tconvs[::-1])
        self.final_linear = nn.Linear(in_ts_length, out_ts_length)

    def forward(self, x):
        res_out = x
        prev_skip = None
        for i in range(self.num_layers):
            res_out, prev_skip = self.dilated_convs[i](res_out, prev_skip)

        for i in range(self.num_layers):
            prev_skip = self.dilated_tconvs[i](prev_skip)
        out = self.final_linear(prev_skip)
        return out


class GraphAttention(torch.nn.Module):
    def __init__(
        self,
        spat_in_channels,
        cl_emb_channels,
        in_ts_length,
        out_ts_length,
        kernel_size,
    ):
        super().__init__()

        # Keys, values and queries
        self.k = torch.nn.Conv2d(
            in_channels=spat_in_channels,  # C
            out_channels=out_ts_length,  # T'
            kernel_size=(1, kernel_size),  # 1, t
            stride=1,
        )
        self.q = torch.nn.Conv2d(
            in_channels=spat_in_channels,
            out_channels=out_ts_length,
            kernel_size=(1, kernel_size),
            stride=1,
        )
        self.v = torch.nn.Conv2d(
            in_channels=spat_in_channels,
            out_channels=out_ts_length,
            kernel_size=(1, kernel_size),
            stride=1,
        )

        # Additional layers
        self.fc_res = torch.nn.Conv2d(
            in_channels=spat_in_channels,  # C
            out_channels=cl_emb_channels,  # C'
            kernel_size=(1, 1),
            stride=1,
        )
        self.fc_res_temp = torch.nn.Conv2d(
            in_channels=in_ts_length,  # C
            out_channels=out_ts_length,  # C'
            kernel_size=(1, 1),
            stride=1,
        )
        self.fc_out = torch.nn.Conv2d(
            in_channels=(in_ts_length - kernel_size) + 1,
            out_channels=cl_emb_channels,  # C'
            kernel_size=(1, 1),
            stride=1,
        )

        # Activation, Normalization and Dropout
        self.act = torch.nn.Softmax(dim=-1)
        self.norm = torch.nn.BatchNorm2d(cl_emb_channels)
        self.dropout = torch.nn.Dropout()

        # To remove
        self.score_style = True

    def forward(self, x):
        k = self.k(x)  # B, T', N, (T - t + 1)
        q = self.q(x)  # B, T', N, (T - t + 1)
        v = self.v(x)  # B, T', N, (T - t + 1)

        score = torch.einsum("BTNC, BTnC -> BTNn", k, q).contiguous()  # B, T', N, N

        score = self.act(score)
        out = torch.einsum(
            "BTnN, BTNC -> BCnT", score, v
        ).contiguous()  # B, (T - t + 1), N, T'

        out = self.fc_out(out)  # B, C', N, T'

        res = self.fc_res(x)  # B, C', N, T
        res = self.fc_res_temp(res.permute(0, 3, 2, 1)) # B, T', N, C'
        res = res.permute(0, 3, 2, 1)

        out = self.norm((out + res))  # B, C', N, T'

        out = self.dropout(out)  # B, C', N, T'

        return out  # B, C', N, T'


class MultiHeadGraphAttention(torch.nn.Module):
    def __init__(
        self,
        spat_in_channels,
        cl_emb_channels,
        in_ts_length,
        out_ts_length,
        ga_kernel_sizes
    ):
        super().__init__()

        # heads
        self.ga = nn.ModuleList()
        for kernel_size in ga_kernel_sizes:
            self.ga.append(GraphAttention(spat_in_channels, cl_emb_channels, in_ts_length, out_ts_length, kernel_size))

        # additional layers
        self.fc_res = torch.nn.Conv2d(
            in_channels=spat_in_channels,
            out_channels=cl_emb_channels,
            kernel_size=(1, 1),
            stride=1,
        )
        self.fc_res_temp = torch.nn.Conv2d(
            in_channels=in_ts_length,
            out_channels=out_ts_length,
            kernel_size=(1, 1),
            stride=1,
        )
        self.fc_out = torch.nn.Conv2d(
            in_channels=cl_emb_channels * len(ga_kernel_sizes),
            out_channels=cl_emb_channels,
            kernel_size=(1, 1),
            stride=1,
        )

        # Normalization and dropout
        self.norm = torch.nn.BatchNorm2d(cl_emb_channels)
        self.dropout = torch.nn.Dropout()

    def forward(self, x):
        res = self.fc_res(x)  # B, C', N, T
        res = self.fc_res_temp(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        ga_out = []
        for ga in self.ga:
            ga_out.append(ga(x))
        x = torch.cat(ga_out, dim=1)  # B, 4*C', N, T'

        x = self.fc_out(x)  # B, C', N, T'

        x = self.norm((x + res))  # B, C', N, T'

        x = self.dropout(x)  # B, C', N, T'

        return x  # B, C', N, T'


class SpatModel(nn.Module):
    def __init__(
        self,
        in_ts_length,
        out_ts_length,
        in_channels,
        out_channels,
        num_ts,
        num_cl,
        cl_emb_channels,
        spat_in_channels,
        ga_kernel_sizes,
    ) -> None:
        super().__init__()

        self.spat_in_conv = nn.Conv2d(in_channels, spat_in_channels, kernel_size=(1, 1))

        # self.cluster_ts = nn.Parameter(torch.randn(num_ts, num_cl))
        # self.uncluster_ts = nn.Parameter(torch.randn(num_cl, num_ts))

        self.mgat = MultiHeadGraphAttention(
            spat_in_channels,
            cl_emb_channels,
            in_ts_length,
            out_ts_length,
            ga_kernel_sizes,
        )
        self.final_linear = nn.Linear(cl_emb_channels, out_channels)

    def forward(self, x):
        # x: B x T x n x c_in
        x = x.permute(0, 3, 2, 1)  # B x c_in x n x T
        spat_in = self.spat_in_conv(x)  # B x c_out x n x T

        # cluster_in = spat_in.permute(0, 1, 3, 2)
        # cluster_out = torch.matmul(cluster_in, self.cluster_ts)  # B x c_out x T x c
        # cluster_out = cluster_out.permute(0, 1, 3, 2)

        mgat_out = self.mgat(spat_in)  # B x c_out x c x T_out

        # uncluster_in = mgat_out.permute(0, 1, 3, 2)
        # uncluster_out = torch.matmul(
        #     uncluster_in, self.uncluster_ts
        # )  # B x c_out x T x n
        # uncluster_out = uncluster_out.permute(0, 1, 3, 2)  # B x c_out x n x T

        mgat_out = mgat_out.permute(0, 3, 2, 1)
        spat_out = self.final_linear(mgat_out)

        return spat_out


class TempModel(nn.Module):
    def __init__(
        self,
        in_ts_length,
        out_ts_length,
        in_channels,
        out_channels,
        temp_num_layers,
        temp_kernel_sizes,
        temp_in_residual_channels,
        temp_out_residual_channels,
        temp_out_skip_channels,
    ) -> None:
        super().__init__()

        self.temp_input_conv = nn.Conv2d(in_channels, temp_in_residual_channels, (1, 1))
        self.temp_conv = nn.ModuleList()
        for num_layer, kernel_size in zip(temp_num_layers, temp_kernel_sizes):
            self.temp_conv.append(
                FDConv2d(
                    in_ts_length,
                    out_ts_length,
                    num_layer,
                    kernel_size,
                    temp_in_residual_channels,
                    temp_out_residual_channels,
                    temp_out_skip_channels,
                )
            )
        self.temp_stack_linear = nn.Linear(len(temp_kernel_sizes), 1)
        self.final_temp_linear = nn.Linear(temp_out_residual_channels, out_channels)

    def forward(self, x):
        # x: B x T x n x c_in
        x = x.permute(0, 3, 2, 1)  # B x c_in x n x T
        temp_in = self.temp_input_conv(x)  # B x c_out x n x T
        temp_outs = []
        for i in range(len(self.temp_conv)):
            temp_out = self.temp_conv[i](temp_in)
            temp_outs.append(temp_out)
        temp_outs = torch.stack(temp_outs, dim=-1)  # B x c_out x n x T x kernel_range
        temp_out = self.temp_stack_linear(temp_outs).squeeze(-1)  # B x c_out x n x T
        temp_out = temp_out.permute(0, 3, 2, 1)  # B x T x n x c_out
        temp_out = self.final_temp_linear(temp_out)  # B x T x n x c_in
        return temp_out


class Model(nn.Module):
    def __init__(
        self,
        lookback_period,
        pred_period,
        in_channels,
        out_channels,
        num_ts,
        num_cl,
        cl_emb_channels,
        temp_num_layers,
        temp_kernel_sizes,
        temp_in_residual_channels,
        temp_out_residual_channels,
        temp_out_skip_channels,
        spat_in_channels,
        ga_kernel_sizes,
        **kwargs,
    ) -> None:
        super().__init__()

        self.temporal_model_1 = TempModel(
            lookback_period,
            pred_period,
            in_channels,
            out_channels,
            temp_num_layers,
            temp_kernel_sizes,
            temp_in_residual_channels,
            temp_out_residual_channels,
            temp_out_skip_channels,
        )
        self.temporal_model_2 = TempModel(
            lookback_period,
            pred_period,
            in_channels,
            out_channels,
            temp_num_layers,
            temp_kernel_sizes,
            temp_in_residual_channels,
            temp_out_residual_channels,
            temp_out_skip_channels,
        )
        self.spat_model = SpatModel(
            lookback_period,
            pred_period,
            in_channels,
            out_channels,
            num_ts,
            num_cl,
            cl_emb_channels,
            spat_in_channels,
            ga_kernel_sizes,
        )
        self.final_linear = nn.Linear(2, 1)
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, hist_ts, fut_ts):
        """
        hist_ts: B x T x n x c_in
        fut_ts: B x P x n x c_in
        """
        # temp_out_gate = self.temporal_model_1(hist_ts)
        # temp_out_gate = torch.sigmoid(temp_out_gate)

        temp_out_reg = self.temporal_model_2(hist_ts)

        spat_out = self.spat_model(hist_ts)

        temp_out = temp_out_reg

        out = torch.stack((temp_out, spat_out), dim=-1)
        out = self.final_linear(out).squeeze(-1)

        return out
