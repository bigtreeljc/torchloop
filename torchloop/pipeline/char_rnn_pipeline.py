from torchloop.util import pipeline_object as po

class char_rnn_pipeline(po.Ipipeline):
    def setup_helper(self):
        ###
        # first get those parameters
        ###
        helper = self.run_helper
        device_name, device_no, dataset_dir, input_dim, output_dim, hidden_dim = \
            helper.device_name, helper.device_no, helper.data_dir, helper.input_dim, \
            helper.output_dim, helper.hidden_dim

        n_layers, batch_size, learning_rate, optimizer, n_iters, print_every, \
        plot_every, matplot, print_formatter_cls, load, train, persist, infer, \
        test, plot = helper.n_layers, helper.batch_size, helper.learning_rate, \
        helper.optimizer, helper.n_iters, helper.print_every, helper.plot_every, \
        helper.matplot, helper.print_formatter_cls, helper.load, helper.train, \
        helper.persist, helper.infer, helper.test, helper.plot 

        ###
        # train loop dataset parameters
        ###
        self.conf.train.dataset.dir = dataset_dir 
        self.conf.train.dataset.batch_size = batch_size
        self.conf.train.dataset.n_iters = n_iters

        ###
        # train loop parameters
        ###
        self.conf.train.train_loop.n_iters = n_iters
        self.conf.train.train_loop.print_every = print_every
        self.conf.train.train_loop.plot_every = plot_every
        self.conf.train.train_loop.matplot = matplot

        ###
        # train loop nn parameters
        ###
        self.conf.train.train_loop.nn.train = train
        self.conf.train.train_loop.nn.input_dim = input_dim
        self.conf.train.train_loop.nn.output_dim = output_dim
        self.conf.train.train_loop.nn.hidden_dim = hidden_dim
        self.conf.train.train_loop.nn.n_layers = n_layers
        self.conf.train.train_loop.nn.load.if_load = load
        self.conf.train.train_loop.nn.persist.if_persist = persist
       
        ###
        # train loop optimizer parameters
        ###
        self.conf.train.train_loop.optimizer.learning_rate = learning_rate
        self.conf.train.train_loop.optimizer.batch_size = batch_size

        ###
        # train loop device parameters
        ###
        self.conf.train.train_loop.device.device_name = device_name
        self.conf.train.train_loop.device.device_no = device_no

        ###
        # infer parameters
        ###
        self.conf.infer.if_infer = infer

        ###
        # test parameters
        ###
        self.conf.test.if_test = test

    def run(self):
        train_o = self.train.trainer_cls(self.conf.train)
        train_o.train()
        # TODO: also add infer and test
